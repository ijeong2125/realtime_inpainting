#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <iostream>
#include <time.h>
#include <vector>
#include <string>
#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <unsupported/Eigen/KroneckerProduct>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

static Eigen::SparseMatrix<double> sparse_vstack(Eigen::SparseMatrix<double> const& upper, Eigen::SparseMatrix<double> const& lower) {
  assert(upper.cols() == lower.cols() && "vstack with mismatching number of columns");

  std::vector<Eigen::Triplet<double, size_t>> triplets;
  triplets.reserve(upper.nonZeros() + lower.nonZeros());

  for (int k = 0; k < upper.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(upper, k); it; ++it) {
      triplets.emplace_back(it.row(), it.col(), it.value());
    }
  }

  for (int k = 0; k < lower.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(lower, k); it; ++it) {
      triplets.emplace_back(upper.rows() + it.row(), it.col(), it.value());
    }
  }

  Eigen::SparseMatrix<double> result(lower.rows() + upper.rows(), upper.cols());
  result.setFromTriplets(triplets.begin(), triplets.end());
  return result;
}

Eigen::SparseMatrix<double> diff(Eigen::SparseMatrix<double> E) {
    Eigen::SparseMatrix<double> E1 = E.block(0, 0, E.rows()-1, E.cols());
    Eigen::SparseMatrix<double> E2 = E.block(1, 0, E.rows()-1, E.cols());
    return E2 - E1;
}

std::vector<int> find(Eigen::VectorXd A){
    std::vector<int> idxs;
    for(int i=0; i<A.size(); ++i)
        if(A(i))
          idxs.push_back(i);
    return idxs;
}

void useBoundary(const Eigen::MatrixXd& img, int &xStart, int &xEnd, int &yStart, int &yEnd) 
{
    int row = img.rows();
    int col = img.cols();
    Eigen::MatrixXd B(row, col);
    B = (img.array() == 0.).cast<double>() * 1.;

    xStart = col - 1;
    xEnd = 0;
    yStart = 0;
    yEnd = 0;

    bool flag = false;
    for (int y = 0; y < row; y++)
        for (int x = 0; x < col; x++)
            if (B(y, x) == 1.)
            {
                if(flag == false)
                {
                    yStart = y;
                    flag = true;
                }
                if (x < xStart)
                    xStart = x;
                if (x > xEnd)
                    xEnd = x;
                yEnd = y;
            }

    bool initialCondition = (xStart == col - 1 && xEnd == 0 && yStart == 0 && yEnd == 0);
    bool noNeedToInpaint = (abs(xStart - xEnd) < 5 && abs(yStart - yEnd) < 5);

    if(initialCondition || noNeedToInpaint)
    {
        xStart = -1;
        return;
    }

    xStart = xStart < 5 ? 0 : xStart - 5;
    yStart = yStart < 5 ? 0 : yStart - 5;
    xEnd = col - xEnd <= 5 ? col : xEnd + 5;
    yEnd = row - yEnd <= 5 ? row : yEnd + 5;
}

image_transport::Publisher img_pub;

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  time_t start,end;
  start = clock();
  cv::Mat img = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8)->image;
  cv::Mat gray;
  cv::cvtColor(img,gray,CV_BGR2GRAY);
  
  Eigen::MatrixXd X_origin;
  cv::cv2eigen(gray,X_origin);
  X_origin = X_origin/255.;
  
  int xStart,xEnd,yStart,yEnd;
  useBoundary(X_origin, xStart, xEnd, yStart, yEnd);
    
  if(xStart==-1)
  {
    sensor_msgs::Image::Ptr img_msg = cv_bridge::CvImage(msg->header,"mono8",gray).toImageMsg();
    img_pub.publish(img_msg);
    end = clock();
    printf("processing time: %lf\n",(double)(end-start)/CLOCKS_PER_SEC);

  }
  else
  {
    Eigen::MatrixXd X;
    X = X_origin.block(yStart,xStart,yEnd-yStart,xEnd-xStart);
    int m = X.rows();
    int n = X.cols();
    int mn = m*n;
    Eigen::MatrixXd B;
    B = (X.array() > 0.).cast<double>()*1.0;
    
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,1>> B_k(B.data(),B.size()); 
    Eigen::VectorXd B_u;
    B_u = Eigen::VectorXd::Ones(mn) - B_k;
    int p = B.sum();
    int q = mn-p;

    Eigen::SparseMatrix<double> eye(m*(n-1),m*(n-1));
    eye.setIdentity();
    Eigen::SparseMatrix<double> zero(m*(n-1),m);
    zero.setZero();
    Eigen::SparseMatrix<double> Dx(m*(n-1),m*(n-1)+m);
    Eigen::SparseMatrix<double> Dx1(m*(n-1),m*(n-1)+m);
    Eigen::SparseMatrix<double> Dx2(m*(n-1),m*(n-1)+m);

    Dx1.leftCols(m) = zero;
    Dx1.rightCols(m*(n-1)) = eye;
    Dx2.leftCols(m*(n-1)) = eye;
    Dx2.rightCols(m) = zero;
    Dx = Dx1-Dx2;

    Eigen::SparseMatrix<double> eye2(n,n);
    eye2.setIdentity();
    Eigen::SparseMatrix<double> eye_diff(m,m);
    eye_diff.setIdentity();
    Eigen::SparseMatrix<double> diff_col(m-1,m);
    diff_col = diff(eye_diff);
    Eigen::SparseMatrix<double> Dy;
    Dy = kroneckerProduct(eye2,diff_col).eval();
    Eigen::SparseMatrix<double> DD(Dx.rows()+Dy.rows(),Dx.cols());
    
    DD = sparse_vstack(Dx, Dy);

    std::vector<int> B_k_idx;
    std::vector<int> B_u_idx;
    B_k_idx = find(B_k);
    B_u_idx = find(B_u);
    
    std::vector<Eigen::Triplet<double>> tripletList_k;
    std::vector<Eigen::Triplet<double>> tripletList_u;
    
    for(int i=0;i<p;i++)
    {
      if(i<q)
      {
        tripletList_u.push_back(Eigen::Triplet<double>(B_u_idx[i],i,1));
      }
      tripletList_k.push_back(Eigen::Triplet<double>(B_k_idx[i],i,1));
    }

    Eigen::SparseMatrix<double> Z_k(mn,p);
    Eigen::SparseMatrix<double> Z_u(mn,q);
    Z_k.setFromTriplets(tripletList_k.begin(),tripletList_k.end());
    Z_u.setFromTriplets(tripletList_u.begin(),tripletList_u.end());

    Eigen::VectorXd x_k;
    Eigen::SparseMatrix<double> A_MATRIX(DD.rows(),p);
    Eigen::VectorXd B_MATRIX;

    x_k = Z_k.transpose() * Eigen::Map<Eigen::VectorXd>(X.data(),X.size());
    A_MATRIX = DD * Z_u;
    B_MATRIX = DD * Z_k * x_k;
    B_MATRIX = -B_MATRIX;

    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::NaturalOrdering<int>> cholesky;
    Eigen::SparseMatrix<double> A(p,p);
    Eigen::VectorXd y;
    Eigen::VectorXd x_u;

    A = A_MATRIX.transpose() * A_MATRIX;
  
    Eigen::VectorXd z; 
    Eigen::VectorXd u;
    Eigen::VectorXd v;
    Eigen::VectorXd b;
    Eigen::VectorXd Ax;
    Eigen::VectorXd rho;

    z = Eigen::VectorXd::Zero(m*(n-1)+(m-1)*n);
    u = Eigen::VectorXd::Zero(m*(n-1)+(m-1)*n);
    
    rho = Eigen::VectorXd::Ones(m*(n-1)+(m-1)*n)/100;
    int iter = 15;

    for(int i=0;i<iter;i++)
    {
      b = B_MATRIX - z + u;
      y = A_MATRIX.transpose() * -b;
      x_u = cholesky.compute(A).solve(y);
      Ax = A_MATRIX * x_u;
      v = Ax + B_MATRIX + u;
      z = (v-rho).cwiseMax(0) - (-v-rho).cwiseMax(0);
      u = u + Ax - z - B_MATRIX;
    }

    Eigen::VectorXd X_recon_vec;    
    X_recon_vec = Z_k*x_k + Z_u*x_u;
    
    Eigen::MatrixXd X_recon;
    X_recon = Eigen::Map<Eigen::MatrixXd>(X_recon_vec.data(),m,n);

    X_origin.block(yStart,xStart,yEnd-yStart,xEnd-xStart) = X_recon; 

    cv::Mat output_gray;
    cv::eigen2cv(X_origin,output_gray);
    cv::Mat output_gray_int;
    output_gray *= (int)255;
    output_gray.convertTo(output_gray_int,CV_8U);

    sensor_msgs::Image::Ptr img_msg = cv_bridge::CvImage(msg->header,"mono8",output_gray_int).toImageMsg();
    img_pub.publish(img_msg);

    end = clock();
    printf("processing time: %lf\n",(double)(end-start)/CLOCKS_PER_SEC);
  }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "inpainting_Eigen");
    ros::NodeHandle nh;    
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber img_sub = it.subscribe("/img_filtered", 1, imageCallback);
    img_pub = it.advertise("/img_reconstructed",1);

    ros::Rate loop_rate(30);
    ros::spin();
    loop_rate.sleep();
    return 0;
}
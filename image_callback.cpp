void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    time_t start,end;
    start = time(NULL); 
    cv::Mat img = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8)->image;
    cv::Mat gray;
    
    int row = img.rows;
    int col = img.cols;

    //image_inpainting using cuda static library
    /////////////////////////////////////////
    //imgArray_recon = wrapper(imageArray);//
    /////////////////////////////////////////
    //end


    //Convert Array to cv::Mat
    uchar *data_B = new uchar[row * col];
    uchar *data_G = new uchar[row * col];
    uchar *data_R = new uchar[row * col];

    uchar *ptr;
    int idx;
    for (int i = 0; i < row; i++){
        ptr = img.ptr<uchar>(i);
        for (int j = 0; j < col; j++){
            idx = i * col + j;
            data_B[idx] = ptr[j * 3 + 0];
            data_G[idx] = ptr[j * 3 + 0];
            data_R[idx] = ptr[j * 3 + 0];
            // printf("%d %d %d\n", data_B[idx], data_G[idx], data_R[idx]);
        }
    }

    cv::Mat output;
    cv::Mat B(row, col, CV_8U, data_B);
    cv::Mat G(row, col, CV_8U, data_G);
    cv::Mat R(row, col, CV_8U, data_R);
    std::vector<cv::Mat> channel{B, G, R};
    cv::merge(channel, output);
    //end
    
    //Publish img_recon_msg
    sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", output).toImageMsg();   
    img_pub.publish(img_msg);

    end = time(NULL);
    double result = end - start;

    //Delete allocated memory 
    delete[] data_B;
    delete[] data_G;
    delete[] data_R;
}

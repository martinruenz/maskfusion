#include <iostream>
#include <iomanip>
#include <map>
#include <fstream>

#include <boost/filesystem.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../Core/Utils/BoundingBox.h"

using namespace cv;
using namespace std;

map<unsigned char, BoundingBox> computeBoundingBoxes(Mat image) {
    if(image.type() != CV_8UC1 || !image.isContinuous()){
        std::cerr << "Invalid input format" << std::endl;
    }
    map<unsigned char, BoundingBox> result;
    for(int r = 0; r < image.rows; r++){
        for(int c = 0; c < image.cols; c++){
            unsigned char v = image.at<unsigned char>(r,c);
            BoundingBox& bb = result[v];
            bb.include(r,c);
        }
    }
    return result;
}

int main(int argc, char * argv[]){
    if(argc < 2) {
        cerr << "Missing parameters." << endl;
        return 1;
    }
    const int index_width = 4;
    for(int i = 0; ; i++){

        // Input
        stringstream path;
        path << argv[1] << "/" << setw(index_width) << setfill('0') << i;
        cv::Mat labels = imread(path.str() + ".png");
        if(labels.total()==0){
            cout << "Stopping at index " << i << "(path not found: " << path.str() << ")" << endl;
            return 0;
        }

        // Only keep single channel image
        if(labels.type() == CV_8UC3){
            Mat t[3];
            split(labels,t);
            labels = t[0];
        }

        // Get bbs
        map<unsigned char, BoundingBox> boxes = computeBoundingBoxes(labels);
        auto l0 = boxes.find(0);
        if(l0 != boxes.end()) boxes.erase(l0);

        // Visualise boxes
        Mat vis;
        equalizeHist(labels, vis);
        for (auto& kv : boxes) kv.second.draw(vis, Scalar(255));
        imshow("boxes", vis);
        waitKey(1);

        // Write output txt files
        ofstream tf(path.str() + ".txt");
        if(argc > 2){
            for(int p=2; p<argc; p++){
                tf << argv[p] << " ";
            }
            tf << "\n";
        }
        for (auto& kv : boxes) {
            BoundingBox& bb = kv.second;
            tf << bb.top << " " << bb.left << " " << bb.bottom << " " << bb.right << endl;
        }
        tf.close();
    }
}

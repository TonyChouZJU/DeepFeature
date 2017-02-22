#include <dirent.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"

#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;
static void save_Mov_Frames(string testImgDirName)
{
    int saveNums = 60;
    string MOV_DirName = "/mnt/exhdd/tomorning_dataset/wonderland/cv/deep_retriver/imgs_retriver/Oss_task_2/" + testImgDirName;
    string trainColor_MOV_DirName = "/mnt/exhdd/tomorning_dataset/wonderland/cv/deep_retriver/imgs_retriver/Oss_retriver_test_2/Train_Color/" + testImgDirName;
    if(access(trainColor_MOV_DirName.c_str(),0) != -1)
        cout << trainColor_MOV_DirName <<"exist!"<<endl;
    else
      mkdir(trainColor_MOV_DirName.c_str(),0777);
    
    string trainGray_MOV_DirName = "/mnt/exhdd/tomorning_dataset/wonderland/cv/deep_retriver/imgs_retriver/Oss_retriver_test_2/Train_Gray/" + testImgDirName;
    if(access(trainGray_MOV_DirName.c_str(),0) != -1)
        cout << trainGray_MOV_DirName <<"exist!"<<endl;
    else
      mkdir(trainGray_MOV_DirName.c_str(),0777);

    string movNamePath;
    vector<Mat> frames; frames.reserve(saveNums);
    vector<Mat> framesGray; framesGray.reserve(saveNums);
    ofstream status_file;
    string status_txt;
    status_txt = trainColor_MOV_DirName + "/status_txt.txt"; 
    status_file.open(status_txt.c_str(), ios::out);

    ofstream Graystatus_file;
    string Graystatus_txt;
    Graystatus_txt = trainGray_MOV_DirName + "/status_txt.txt"; 
    Graystatus_file.open(Graystatus_txt.c_str(), ios::out);
    for(int j= 0; j<8; j++)
    {
        movNamePath = MOV_DirName;
        stringstream ss;ss <<j;
        string str_j; ss>>str_j;
        movNamePath = movNamePath + "/" + testImgDirName + '_' +str_j + ".mov";
        cout <<"MovNamePath::::::"<<movNamePath<<endl;
        if(access(movNamePath.c_str(),0) != -1)
        {
            status_file << testImgDirName <<"_" <<str_j <<".mov============";
            Graystatus_file << testImgDirName <<"_" <<str_j <<".mov============";
            VideoCapture seq(movNamePath);
            int k;
            for(k=0;;k++)
            {
                Mat frame,color,gray;
                seq>>frame;
                if(frame.empty())   break;
                transpose(frame,frame);
                flip(frame,frame,1);
                //resize(frame,color,Size(120,160));
                color = frame;
                cvtColor(color,gray, CV_BGR2GRAY); 
                frames.push_back(color);
                framesGray.push_back(gray);
            }
            status_file << "size:"<<k+1<<endl;
            Graystatus_file << "size:"<<k+1<<endl;
        }
    }
    int gap_size = frames.size()/saveNums;
    //左上
    for(int i = 1; i<=saveNums; i++)
    {
        int idx = (i-1)*gap_size;
        stringstream ss; ss<<i;
        string str_i; ss>>str_i;
        string tmpColorFrameName = trainColor_MOV_DirName + "/" + testImgDirName+"_"+str_i + ".jpg";
        string tmpGrayFrameName = trainGray_MOV_DirName + "/" + testImgDirName+"_"+str_i + ".jpg";

        Mat imgColor = frames[idx];
        Mat imgGray = framesGray[idx];
        
        Mat Color = imgColor(Rect(0,0,450,600));
        Mat roiColor; resize(Color, roiColor, Size(256,256));


        Mat Gray = imgGray(Rect(0,0,450, 600));
        Mat roiGray; resize(Gray, roiGray, Size(256,256));

        if(!imwrite(tmpColorFrameName, roiColor))
        {
            cout <<"cannot save Color Frame " <<i <<endl;
            continue;
        }
        if(!imwrite(tmpGrayFrameName, roiGray))
        {
            cout <<"cannot save Gray Frame " <<i <<endl;
            continue;
        }
        status_file <<idx<<endl;
        Graystatus_file <<idx<<endl;
    }
//右上
    for(int i = 1; i<=saveNums; i++)
    {
        int idx = (i-1)*gap_size;
        stringstream ss; ss<<i + saveNums*1;
        string str_i; ss>>str_i;
        string tmpColorFrameName = trainColor_MOV_DirName + "/" + testImgDirName+"_"+str_i + ".jpg";
        string tmpGrayFrameName = trainGray_MOV_DirName + "/" + testImgDirName+"_"+str_i + ".jpg";

        Mat imgColor = frames[idx];
        Mat imgGray = framesGray[idx];
        
        Mat Color = imgColor(Rect(30,0,450,600));
        Mat roiColor; resize(Color, roiColor, Size(256,256));


        Mat Gray = imgGray(Rect(30,0,450, 600));
        Mat roiGray; resize(Gray, roiGray, Size(256,256));

        if(!imwrite(tmpColorFrameName, roiColor))
        {
            cout <<"cannot save Color Frame " <<i <<endl;
            continue;
        }
        if(!imwrite(tmpGrayFrameName, roiGray))
        {
            cout <<"cannot save Gray Frame " <<i <<endl;
            continue;
        }
        status_file <<idx<<endl;
        Graystatus_file <<idx<<endl;
    }
//左下
    for(int i = 1; i<=saveNums; i++)
    {
        int idx = (i-1)*gap_size;
        stringstream ss; ss<<i + saveNums*2;
        string str_i; ss>>str_i;
        string tmpColorFrameName = trainColor_MOV_DirName + "/" + testImgDirName+"_"+str_i + ".jpg";
        string tmpGrayFrameName = trainGray_MOV_DirName + "/" + testImgDirName+"_"+str_i + ".jpg";

        Mat imgColor = frames[idx];
        Mat imgGray = framesGray[idx];
        
        Mat Color = imgColor(Rect(0,40,450,600));
        Mat roiColor; resize(Color, roiColor, Size(256,256));


        Mat Gray = imgGray(Rect(0,40,450, 600));
        Mat roiGray; resize(Gray, roiGray, Size(256,256));

        if(!imwrite(tmpColorFrameName, roiColor))
        {
            cout <<"cannot save Color Frame " <<i <<endl;
            continue;
        }
        if(!imwrite(tmpGrayFrameName, roiGray))
        {
            cout <<"cannot save Gray Frame " <<i <<endl;
            continue;
        }
        status_file <<idx<<endl;
        Graystatus_file <<idx<<endl;
    }
//右下
    for(int i = 1; i<=saveNums; i++)
    {
        int idx = (i-1)*gap_size;
        stringstream ss; ss<<i + saveNums*3;
        string str_i; ss>>str_i;
        string tmpColorFrameName = trainColor_MOV_DirName + "/" + testImgDirName+"_"+str_i + ".jpg";
        string tmpGrayFrameName = trainGray_MOV_DirName + "/" + testImgDirName+"_"+str_i + ".jpg";

        Mat imgColor = frames[idx];
        Mat imgGray = framesGray[idx];
        
        Mat Color = imgColor(Rect(30,40,450,600));
        Mat roiColor; resize(Color, roiColor, Size(256,256));


        Mat Gray = imgGray(Rect(30,40,450, 600));
        Mat roiGray; resize(Gray, roiGray, Size(256,256));

        if(!imwrite(tmpColorFrameName, roiColor))
        {
            cout <<"cannot save Color Frame " <<i <<endl;
            continue;
        }
        if(!imwrite(tmpGrayFrameName, roiGray))
        {
            cout <<"cannot save Gray Frame " <<i <<endl;
            continue;
        }
        status_file <<idx<<endl;
        Graystatus_file <<idx<<endl;
    }
//中心
    for(int i = 1; i<=saveNums; i++)
    {
        int idx = (i-1)*gap_size;
        stringstream ss; ss<<i + saveNums*4;
        string str_i; ss>>str_i;
        string tmpColorFrameName = trainColor_MOV_DirName + "/" + testImgDirName+"_"+str_i + ".jpg";
        string tmpGrayFrameName = trainGray_MOV_DirName + "/" + testImgDirName+"_"+str_i + ".jpg";

        Mat imgColor = frames[idx];
        Mat imgGray = framesGray[idx];
        
        Mat Color = imgColor(Rect(15,20,450,600));
        Mat roiColor; resize(Color, roiColor, Size(256,256));


        Mat Gray = imgGray(Rect(15,20,450, 600));
        Mat roiGray; resize(Gray, roiGray, Size(256,256));

        if(!imwrite(tmpColorFrameName, roiColor))
        {
            cout <<"cannot save Color Frame " <<i <<endl;
            continue;
        }
        if(!imwrite(tmpGrayFrameName, roiGray))
        {
            cout <<"cannot save Gray Frame " <<i <<endl;
            continue;
        }
        status_file <<idx<<endl;
        Graystatus_file <<idx<<endl;
    }
//原始
    for(int i = 1; i<=saveNums; i++)
    {
        int idx = (i-1)*gap_size;
        stringstream ss; ss<<i + saveNums*5;
        string str_i; ss>>str_i;
        string tmpColorFrameName = trainColor_MOV_DirName + "/" + testImgDirName+"_"+str_i + ".jpg";
        string tmpGrayFrameName = trainGray_MOV_DirName + "/" + testImgDirName+"_"+str_i + ".jpg";

        Mat imgColor = frames[idx];
        Mat imgGray = framesGray[idx];
        
        Mat Color = imgColor;
        Mat roiColor; resize(Color, roiColor, Size(256,256));


        Mat Gray = imgGray;
        Mat roiGray; resize(Gray, roiGray, Size(256,256));

        if(!imwrite(tmpColorFrameName, roiColor))
        {
            cout <<"cannot save Color Frame " <<i <<endl;
            continue;
        }
        if(!imwrite(tmpGrayFrameName, roiGray))
        {
            cout <<"cannot save Gray Frame " <<i <<endl;
            continue;
        }
        status_file <<idx<<endl;
        Graystatus_file <<idx<<endl;
    }

    status_file.close();
    Graystatus_file.close();
}

static void save_testImg_dirName( string &test_img_dir)
{
	DIR *dp_img;
	struct dirent *dirp_img;
    vector<string> imgs_dir_name_vec;

	if((dp_img = opendir(test_img_dir.c_str()))==NULL)
		cout <<"Can't open"<<test_img_dir<<endl;

	while((dirp_img = readdir(dp_img)) != NULL)
    	if(dirp_img->d_type == 4)   //directory, not file
    	{
            if(strcmp(dirp_img->d_name,".")==0 || strcmp(dirp_img->d_name,"..")==0 ) 
                continue;
	        string imgDirName(dirp_img->d_name);
            save_Mov_Frames(imgDirName);
            imgs_dir_name_vec.push_back(imgDirName);
    	}
	closedir(dp_img);
}

int main(int argc, char** argv)
{
    string test_img_dir(argv[1]);
    save_testImg_dirName(test_img_dir);
    return 0;
}

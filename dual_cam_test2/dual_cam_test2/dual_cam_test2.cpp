#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <stdarg.h>
#include <time.h>

// 单窗口显示多幅图像的函数
void cvShowMultiImages(char* title, int nArgs, ...)
{
	// 略，详见学习笔记（5）
}


int main(int argc, char** argv)
{
	int cam_count;

	//仅仅获取摄像头数目
	cam_count = CCameraDS::CameraCount();
	printf("There are %d cameras./n", cam_count);


	//获取所有摄像头的名称
	for (int i = 0; i < cam_count; i++)
	{
		char camera_name[1024];
		int retval = CCameraDS::CameraName(i, camera_name, sizeof(camera_name));

		if (retval >0)
			printf("Camera #%d's Name is '%s'./n", i, camera_name);
		else
			printf("Can not get Camera #%d's name./n", i);
	}

	if (cam_count == 0)
		return -1;

	// 创建2个摄像头类
	CCameraDS camera1;
	CCameraDS camera2;

	//打开第一个摄像头
	//if(! camera.OpenCamera(0, true)) //弹出属性选择窗口
	if (!camera1.OpenCamera(0, false, 320, 240)) //不弹出属性选择窗口，用代码制定图像宽和高
	{
		fprintf(stderr, "Can not open camera./n");
		return -1;
	}
	//打开第二个摄像头
	camera2.OpenCamera(1, false, 320, 240);


	cvNamedWindow("Multiple Cameras");

	// 初始化在子图像中显示字符的字体格式
	CvFont tFont;
	cvInitFont(&tFont, CV_FONT_HERSHEY_COMPLEX, 0.5f, 0.7f, 0, 1, 8);

	char cam1str[] = "Camera #1";
	char cam2str[] = "Camera #2";

	// 为读取系统时间信息分配内存
	char timestr[25];
	memset(timestr, 0, 25 * sizeof(char));

	while (1)
	{
		//获取一帧
		IplImage *pFrame1 = camera1.QueryFrame();
		IplImage *pFrame2 = camera2.QueryFrame();

		// 获取当前帧的灰度图
		IplImage* frame_gray_1 = cvCreateImage(cvGetSize(pFrame1), pFrame1->depth, 1);
		IplImage* frame_gray_2 = cvCreateImage(cvGetSize(pFrame2), pFrame2->depth, 1);
		cvCvtColor(pFrame1, frame_gray_1, CV_RGB2GRAY);
		cvCvtColor(pFrame2, frame_gray_2, CV_RGB2GRAY);

		// 对灰度图像进行Canny边缘检测
		// 然后将图像通道数改为三通道
		IplImage* frame_canny_1 = cvCreateImage(cvGetSize(pFrame1), pFrame1->depth, 1);
		IplImage* frame_canny_2 = cvCreateImage(cvGetSize(pFrame2), pFrame2->depth, 1);
		IplImage* frame1 = cvCreateImage(cvGetSize(pFrame1), pFrame1->depth, pFrame1->nChannels);
		IplImage* frame2 = cvCreateImage(cvGetSize(pFrame2), pFrame2->depth, pFrame2->nChannels);
		cvCanny(frame_gray_1, frame_canny_1, 20, 75, 3);
		cvCanny(frame_gray_2, frame_canny_2, 20, 75, 3);
		cvCvtColor(frame_canny_1, frame1, CV_GRAY2BGR);
		cvCvtColor(frame_canny_2, frame2, CV_GRAY2BGR);


		// 获取系统时间信息
		time_t rawtime;
		struct tm* timeinfo;

		rawtime = time(NULL);
		timeinfo = localtime(&rawtime);
		char* p = asctime(timeinfo);

		// 字符串 p 的第25个字符是换行符 '/n'
		// 但在子图像中将乱码显示
		// 故仅读取 p 的前 24 个字符
		for (int i = 0; i < 24; i++)
		{
			timestr[i] = *p;
			p++;
		}
		p = NULL;

		// 在每个子图像上显示摄像头序号以及系统时间信息
		cvPutText(pFrame1, cam1str, cvPoint(95, 15), &tFont, CV_RGB(255, 0, 0));
		cvPutText(pFrame2, cam2str, cvPoint(95, 15), &tFont, CV_RGB(255, 0, 0));
		cvPutText(frame1, cam1str, cvPoint(95, 15), &tFont, CV_RGB(255, 0, 0));
		cvPutText(frame2, cam2str, cvPoint(95, 15), &tFont, CV_RGB(255, 0, 0));

		cvPutText(pFrame1, timestr, cvPoint(5, 225), &tFont, CV_RGB(255, 0, 0));
		cvPutText(pFrame2, timestr, cvPoint(5, 225), &tFont, CV_RGB(255, 0, 0));
		cvPutText(frame1, timestr, cvPoint(5, 225), &tFont, CV_RGB(255, 0, 0));
		cvPutText(frame2, timestr, cvPoint(5, 225), &tFont, CV_RGB(255, 0, 0));

		// 显示实时的摄像头视频
		cvShowMultiImages("Multiple Cameras", 4, pFrame1, pFrame2, frame1, frame2);


		//cvWaitKey(33);
		int key = cvWaitKey(33);
		if (key == 27) break;

		cvReleaseImage(&frame1);
		cvReleaseImage(&frame2);
		cvReleaseImage(&frame_gray_1);
		cvReleaseImage(&frame_gray_2);
		cvReleaseImage(&frame_canny_1);
		cvReleaseImage(&frame_canny_2);
	}

	camera1.CloseCamera(); //可不调用此函数，CCameraDS析构时会自动关闭摄像头
	camera2.CloseCamera();

	cvDestroyWindow("Multiple Cameras");

	return 0;
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <stdarg.h>
#include <time.h>

// ��������ʾ���ͼ��ĺ���
void cvShowMultiImages(char* title, int nArgs, ...)
{
	// �ԣ����ѧϰ�ʼǣ�5��
}


int main(int argc, char** argv)
{
	int cam_count;

	//������ȡ����ͷ��Ŀ
	cam_count = CCameraDS::CameraCount();
	printf("There are %d cameras./n", cam_count);


	//��ȡ��������ͷ������
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

	// ����2������ͷ��
	CCameraDS camera1;
	CCameraDS camera2;

	//�򿪵�һ������ͷ
	//if(! camera.OpenCamera(0, true)) //��������ѡ�񴰿�
	if (!camera1.OpenCamera(0, false, 320, 240)) //����������ѡ�񴰿ڣ��ô����ƶ�ͼ���͸�
	{
		fprintf(stderr, "Can not open camera./n");
		return -1;
	}
	//�򿪵ڶ�������ͷ
	camera2.OpenCamera(1, false, 320, 240);


	cvNamedWindow("Multiple Cameras");

	// ��ʼ������ͼ������ʾ�ַ��������ʽ
	CvFont tFont;
	cvInitFont(&tFont, CV_FONT_HERSHEY_COMPLEX, 0.5f, 0.7f, 0, 1, 8);

	char cam1str[] = "Camera #1";
	char cam2str[] = "Camera #2";

	// Ϊ��ȡϵͳʱ����Ϣ�����ڴ�
	char timestr[25];
	memset(timestr, 0, 25 * sizeof(char));

	while (1)
	{
		//��ȡһ֡
		IplImage *pFrame1 = camera1.QueryFrame();
		IplImage *pFrame2 = camera2.QueryFrame();

		// ��ȡ��ǰ֡�ĻҶ�ͼ
		IplImage* frame_gray_1 = cvCreateImage(cvGetSize(pFrame1), pFrame1->depth, 1);
		IplImage* frame_gray_2 = cvCreateImage(cvGetSize(pFrame2), pFrame2->depth, 1);
		cvCvtColor(pFrame1, frame_gray_1, CV_RGB2GRAY);
		cvCvtColor(pFrame2, frame_gray_2, CV_RGB2GRAY);

		// �ԻҶ�ͼ�����Canny��Ե���
		// Ȼ��ͼ��ͨ������Ϊ��ͨ��
		IplImage* frame_canny_1 = cvCreateImage(cvGetSize(pFrame1), pFrame1->depth, 1);
		IplImage* frame_canny_2 = cvCreateImage(cvGetSize(pFrame2), pFrame2->depth, 1);
		IplImage* frame1 = cvCreateImage(cvGetSize(pFrame1), pFrame1->depth, pFrame1->nChannels);
		IplImage* frame2 = cvCreateImage(cvGetSize(pFrame2), pFrame2->depth, pFrame2->nChannels);
		cvCanny(frame_gray_1, frame_canny_1, 20, 75, 3);
		cvCanny(frame_gray_2, frame_canny_2, 20, 75, 3);
		cvCvtColor(frame_canny_1, frame1, CV_GRAY2BGR);
		cvCvtColor(frame_canny_2, frame2, CV_GRAY2BGR);


		// ��ȡϵͳʱ����Ϣ
		time_t rawtime;
		struct tm* timeinfo;

		rawtime = time(NULL);
		timeinfo = localtime(&rawtime);
		char* p = asctime(timeinfo);

		// �ַ��� p �ĵ�25���ַ��ǻ��з� '/n'
		// ������ͼ���н�������ʾ
		// �ʽ���ȡ p ��ǰ 24 ���ַ�
		for (int i = 0; i < 24; i++)
		{
			timestr[i] = *p;
			p++;
		}
		p = NULL;

		// ��ÿ����ͼ������ʾ����ͷ����Լ�ϵͳʱ����Ϣ
		cvPutText(pFrame1, cam1str, cvPoint(95, 15), &tFont, CV_RGB(255, 0, 0));
		cvPutText(pFrame2, cam2str, cvPoint(95, 15), &tFont, CV_RGB(255, 0, 0));
		cvPutText(frame1, cam1str, cvPoint(95, 15), &tFont, CV_RGB(255, 0, 0));
		cvPutText(frame2, cam2str, cvPoint(95, 15), &tFont, CV_RGB(255, 0, 0));

		cvPutText(pFrame1, timestr, cvPoint(5, 225), &tFont, CV_RGB(255, 0, 0));
		cvPutText(pFrame2, timestr, cvPoint(5, 225), &tFont, CV_RGB(255, 0, 0));
		cvPutText(frame1, timestr, cvPoint(5, 225), &tFont, CV_RGB(255, 0, 0));
		cvPutText(frame2, timestr, cvPoint(5, 225), &tFont, CV_RGB(255, 0, 0));

		// ��ʾʵʱ������ͷ��Ƶ
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

	camera1.CloseCamera(); //�ɲ����ô˺�����CCameraDS����ʱ���Զ��ر�����ͷ
	camera2.CloseCamera();

	cvDestroyWindow("Multiple Cameras");

	return 0;
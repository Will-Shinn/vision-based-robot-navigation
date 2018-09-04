#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <opencv/cxcore.h>
using namespace cv;
using namespace std;
class FilterEngine
{
	public:
		// �յĹ��캯�� 
		FilterEngine();
		// ����2D�Ĳ��ɷֵ��˲���(!_filter2D.empty())����
		// �ɷֵ��˲�
		//��(!_rowFilter.empty() && !_columnFilter.empty())
		// ������������Ϊ "srcType", �������Ϊ"dstType",
		// �м����������Ϊ "bufType".
		// _rowBorderType �� _columnBorderType ����ͼ��߽���α��� ������ 
		// ֻ�� _rowBorderType and/or _columnBorderType
		// == BORDER_CONSTANT ʱ _borderValue �Żᱻ�õ�
		FilterEngine(const Ptr<BaseFilter>& _filter2D,
			const Ptr<BaseRowFilter>& _rowFilter,
			int srcType, int dstType, int bufType,
			int _rowBorderType = BORDER_REPLICATE,
			int _columnBorderType = -1, // Ĭ��ʹ�� _rowBorderType
			const Scalar& _borderValue = Scalar());
			virtual ~FilterEngine();
			// ��ʼ����ķָ��
			void init(const Ptr<BaseFilter>& _filter2D,
			const Ptr<BaseRowFilter>& _rowFilter,
			const Ptr<BaseColumnFilter>& _columnFilter,
			int srcType, int dstType, int bufType,
			int _rowBorderType = BORDER_REPLICATE, int _columnBorderType = -1,
			const Scalar& _borderValue = Scalar());
			// ����ͼ��ߴ�"wholeSize"ΪROI��ʼ�˲�.   29.    // ����ͼ��ʼ��y-position����.
			virtual int start(Size wholeSize, Rect roi, int maxBufRows = -1);// ��һ����Ҫͼ��Ŀ�ʼ
			virtual int start(const Mat& src, const Rect& srcRoi = Rect(0, 0, -1, -1),
			bool isolated = false, int maxBufRows = -1);
			// ����Դͼ�����һ���� 
			// ��"src"��"dst"����"srcCount" ��
			// ���ش��������
			virtual int proceed(const uchar* src, int srcStep, int srcCount,
				uchar* dst, int dstStep);
			// ��������ROI�ĸ߲����
			virtual void apply(const Mat& src, Mat& dst,
				const Rect& srcRoi = Rect(0, 0, -1, -1),
				Point dstOfs = Point(0, 0),
				bool isolated = false);
			bool isSeparable() const { return filter2D.empty(); }
			// ����ͼ��δ�����������
			int remainingInputRows() const;
			// ������δ�����������
			int remainingOutputRows() const;
			// Դͼ�Ŀ�ʼ�ͽ�����
			int startY, endY;
			// ָ���˲�����ָ��
			Ptr<BaseFilter> filter2D;
			Ptr<BaseRowFilter> rowFilter;
			Ptr<BaseColumnFilter> columnFilter;
};
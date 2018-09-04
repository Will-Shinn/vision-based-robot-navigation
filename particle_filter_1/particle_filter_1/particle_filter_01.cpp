#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <opencv/cxcore.h>
using namespace cv;
using namespace std;
class FilterEngine
{
	public:
		// 空的构造函数 
		FilterEngine();
		// 构造2D的不可分的滤波器(!_filter2D.empty())或者
		// 可分的滤波
		//器(!_rowFilter.empty() && !_columnFilter.empty())
		// 输入数据类型为 "srcType", 输出类型为"dstType",
		// 中间的数据类型为 "bufType".
		// _rowBorderType 何 _columnBorderType 决定图像边界如何被外 推扩充 
		// 只有 _rowBorderType and/or _columnBorderType
		// == BORDER_CONSTANT 时 _borderValue 才会被用到
		FilterEngine(const Ptr<BaseFilter>& _filter2D,
			const Ptr<BaseRowFilter>& _rowFilter,
			int srcType, int dstType, int bufType,
			int _rowBorderType = BORDER_REPLICATE,
			int _columnBorderType = -1, // 默认使用 _rowBorderType
			const Scalar& _borderValue = Scalar());
			virtual ~FilterEngine();
			// 初始引擎的分割函数
			void init(const Ptr<BaseFilter>& _filter2D,
			const Ptr<BaseRowFilter>& _rowFilter,
			const Ptr<BaseColumnFilter>& _columnFilter,
			int srcType, int dstType, int bufType,
			int _rowBorderType = BORDER_REPLICATE, int _columnBorderType = -1,
			const Scalar& _borderValue = Scalar());
			// 定义图像尺寸"wholeSize"为ROI开始滤波.   29.    // 返回图像开始的y-position坐标.
			virtual int start(Size wholeSize, Rect roi, int maxBufRows = -1);// 另一种需要图像的开始
			virtual int start(const Mat& src, const Rect& srcRoi = Rect(0, 0, -1, -1),
			bool isolated = false, int maxBufRows = -1);
			// 处理源图像的另一部分 
			// 从"src"到"dst"处理"srcCount" 行
			// 返回处理的行数
			virtual int proceed(const uchar* src, int srcStep, int srcCount,
				uchar* dst, int dstStep);
			// 处理整个ROI的高层调用
			virtual void apply(const Mat& src, Mat& dst,
				const Rect& srcRoi = Rect(0, 0, -1, -1),
				Point dstOfs = Point(0, 0),
				bool isolated = false);
			bool isSeparable() const { return filter2D.empty(); }
			// 输入图中未被处理的行数
			int remainingInputRows() const;
			// 输入中未被处理的行数
			int remainingOutputRows() const;
			// 源图的开始和结束行
			int startY, endY;
			// 指向滤波器的指针
			Ptr<BaseFilter> filter2D;
			Ptr<BaseRowFilter> rowFilter;
			Ptr<BaseColumnFilter> columnFilter;
};
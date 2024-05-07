#include "opencv2\core.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\highgui\highgui.hpp"
#include <iostream>
#include <string>
#include <cstdlib>
#include <cmath>
#include <stack>

using namespace std;
using namespace cv;

#define MAX_FILTER_COLS 5
#define PI 3.1415926535897932384626433832795

// ham loc anh cap xam qua mat na filter co so cot toi da la 5 va tra ve ma tran kieu int
void Filter(const Mat src, Mat & dst, int filter[][MAX_FILTER_COLS], int filterRows, int filterCols) {

	// sao chep anh nguon
	Mat image = src.clone();
	// tao ma tran diem anh kich thuoc bang anh nguon
	dst = Mat_<int>(src.rows, src.cols);
	
	int temp;
	int x, y;
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			temp = 0;
			for (int m = 0; m < filterRows; m++) {
				for (int n = 0; n < filterCols; n++) {
					x = i + m - 1;
					y = j + n - 1;
					// kiem tra x, y co nam trong anh khong va loc diem anh
					if(0 <= x && x < image.rows && 0 <= y && y < image.cols) {
						temp += image.at<uchar>(x, y) * filter[m][n];
					}
				}
			}
			// luu diem anh vua duoc loc
			dst.at<int>(i, j) = temp;
		}
	}
}

// tim bien bang Prewitt
int detectByPrewitt(Mat src, Mat& dst) {
	// bo loc theo X va Y
	int filterX[3][MAX_FILTER_COLS] = {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}};
	int filterY[3][MAX_FILTER_COLS] = { {-1, -1, -1}, {0, 0, 0}, {1, 1, 1} };

	// loc anh theo mat na x, y
	Mat X, Y;
	Filter(src, X, filterX, 3, 3);
	Filter(src, Y, filterY, 3, 3);
	
	// ghep hai dao ham anh
	Mat PrewittX, PrewittY;
	dst = Mat(src.rows, src.cols, src.type());
	PrewittX = Mat(src.rows, src.cols, src.type());
	PrewittY = Mat(src.rows, src.cols, src.type());

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			PrewittX.at<uchar>(i, j) = saturate_cast<uchar>(abs(X.at<int>(i, j)));
			PrewittY.at<uchar>(i, j) = saturate_cast<uchar>(abs(Y.at<int>(i, j)));
			dst.at<uchar>(i, j) = saturate_cast<uchar>(PrewittX.at<uchar>(i, j) + PrewittY.at<uchar>(i, j));
		}
	}

	// hien thi anh loc theo x, y
	namedWindow("Prewitt X", WINDOW_AUTOSIZE);
	imshow("Prewitt X", PrewittX);

	namedWindow("Prewitt Y", WINDOW_AUTOSIZE);
	imshow("Prewitt Y", PrewittY);

	return 0;
}

// tim bien bang Sobel
int detectBySobel(Mat src, Mat& dst) {
	// bo loc theo X va Y
	int filterX[3][MAX_FILTER_COLS] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
	int filterY[3][MAX_FILTER_COLS] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

	// loc anh theo mat na x, y
	Mat X, Y;
	Filter(src, X, filterX, 3, 3);
	Filter(src, Y, filterY, 3, 3);

	// ghep hai dao ham anh
	Mat SobelX, SobelY;
	dst = Mat(src.rows, src.cols, src.type());
	SobelX = Mat(src.rows, src.cols, src.type());
	SobelY = Mat(src.rows, src.cols, src.type());

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			SobelX.at<uchar>(i, j) = saturate_cast<uchar>(abs(X.at<int>(i, j)));
			SobelY.at<uchar>(i, j) = saturate_cast<uchar>(abs(Y.at<int>(i, j)));
			dst.at<uchar>(i, j) = saturate_cast<uchar>(SobelX.at<uchar>(i, j) + SobelY.at<uchar>(i, j));
		}
	}

	// hien thi anh loc theo x, y

	namedWindow("Sobel X", WINDOW_AUTOSIZE);
	imshow("Sobel X", SobelX);

	namedWindow("Sobel Y", WINDOW_AUTOSIZE);
	imshow("Sobel Y", SobelY);

	return 0;
}

int detectByLaplace(Mat src, Mat& dst) {
	// mat na dao ham cap 2
	int filter[3][MAX_FILTER_COLS] = { {-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1} };

	// loc anh theo mat na
	Mat image;
	Filter(src, image, filter, 3, 3);

	dst = Mat(src.rows, src.cols, src.type());
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			dst.at<uchar>(i, j) = saturate_cast<uchar>(image.at<int>(i, j));
		}
	}

	return 0;
}

// ham lam min anh, giam nhieu
void Smoothing(const Mat& src, Mat& dst) {
	// mat na lam min anh
	
	int  GaussianFilter[5][MAX_FILTER_COLS] = { {2, 4, 5, 4, 2},
												{4, 9, 12, 9, 4},
												{5, 12, 15, 12, 5},
												{4, 9, 12, 9, 4},
												{2, 4, 5, 4, 2} };
	

	Mat image;
	Filter(src, image, GaussianFilter, 5, 5);
	dst = Mat_<uchar>(src.rows, src.cols);

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			dst.at<uchar>(i, j) = saturate_cast<uchar>(image.at<int>(i, j) / 159);
		}
	}
}

// so sanh diem anh vi tri x, y voi diem anh vi tri vuong goc
uchar valueNon_MaximumSuppression(const Mat& image, int x, int y, int dx, int dy) {
	for (int i = 1; i >= -1; i -= 2) {
		dx = dx * i;
		dy = dy * i;
		if (0 <= x + dx && x + dx < image.rows && 0 <= y + dy && y + dy < image.cols) {
			if (image.at<int>(x, y) <= image.at<int>(x + dx, y + dy)) {
				return 0;
			}
		}
	}
	return saturate_cast<uchar>(image.at<int>(x, y));
}


// ham tinh toan do doc, huong sang, va thuc hien chon diem ung vien bien canh
void GradientComputation_And_Non_MaximumSuppression(const Mat& src, Mat& dst) {
	Mat image = Mat_<int>(src.rows, src.cols);


	// tinh toan do doc
	// dao ham 2 anh theo 2 huong X, Y
	int filterX[3][MAX_FILTER_COLS] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
	int filterY[3][MAX_FILTER_COLS] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };
	Mat GX, GY;
	Filter(src, GX, filterX, 3, 3);
	Filter(src, GY, filterY, 3, 3);

	// ghep dao ham ai anh
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			image.at<int>(i, j) = abs(GX.at<int>(i, j)) + abs(GY.at<int>(i, j));
		}
	}


	// Non-Maximum Suppression
	// chon diem ung vien bien canh
	double EdgeDirection; // huong canh

	dst = Mat_<uchar>(src.rows, src.cols);
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			// tinh goc huong cua canh
			EdgeDirection = atan2(GY.at<int>(i, j), GX.at<int>(i, j)) * 180 / PI;
			// doi goc ve trong khoang 0 -> 180 do
			if (EdgeDirection < 0) {
				EdgeDirection += 180;
			}
			
			dst.at<uchar>(i, j) = 0;

			// 0 +- 22.5
			if (157.5 <= EdgeDirection || EdgeDirection < 22.5) {
				dst.at<uchar>(i, j) = valueNon_MaximumSuppression(image, i, j, 0, 1);
			}
			// 45 +- 22.5
			else if (22.5 <= EdgeDirection && EdgeDirection < 67.5) {
				dst.at<uchar>(i, j) = valueNon_MaximumSuppression(image, i, j, 1, 1);
			}
			// 90 +-22.5
			else if (67.5 <= EdgeDirection && EdgeDirection < 112.5) {
				dst.at<uchar>(i, j) = valueNon_MaximumSuppression(image, i, j, 1, 0);
			}
			// 135 +- 22.5
			else if (112.5 <= EdgeDirection && EdgeDirection < 157.5) {
				dst.at<uchar>(i, j) = valueNon_MaximumSuppression(image, i, j, -1, 1);
			};
		}
	}
}

// chon cac diem bien canh dua tren nguong cao, nguong thap
void Hysteresis(const Mat& src, Mat& dst, uchar lowThreshold, uchar highThreshold) {
	Mat image = src.clone();
	dst = Mat_<uchar>(src.rows, src.cols);

	struct Pixel {
		int x, y;
	};
	stack<Pixel>edgePixels; // luu nhung diem chac chan la canh
	Pixel pixel;
	Mat visited = Mat_<bool>(image.rows, image.cols); // mang danh dau diem da them roi

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			visited.at<bool>(i, j) = false;
			dst.at<uchar>(i, j) = 0;
			// lay cac diem tren nguong cao
			if (image.at<uchar>(i, j) > highThreshold) {
				edgePixels.push({ i, j });
				visited.at<bool>(i, j) = true; // danh dau da lay
			}
		}
	}

	// lay cac diem tren nguong thap canh cac diem duoc chon
	while (!edgePixels.empty()) {
		pixel = edgePixels.top();
		edgePixels.pop();
		// lay diem bien canh
		dst.at<uchar>(pixel.x, pixel.y) = 255;


		// kiem tra cac diem lan can
		int x, y;

		for (int dx = -1; dx <= 1; dx++) {
			for (int dy = -1; dy <= 1; dy++) {
				x = pixel.x + dx;
				y = pixel.y + dy;
				// kiem tra vi tri x, y co trong ma tran khong
				if (0 <= x && x < image.rows && 0 <= y && y < image.cols) {
					// neu diem lan can chua lay va diem do co muc sang cao hon nguong thap thi lay
					if (!visited.at<bool>(x, y) && image.at<uchar>(x, y) > lowThreshold) {
						visited.at<bool>(x, y) = true;
						edgePixels.push({ x, y });
					}
				}
			}
		}
	}

}

// ham lay bien canh ban thuat Canny
int detectByCany(Mat src, Mat & dst, uchar lowThreshold = 50, uchar highThreshold = 150) {

	// lam min anh
	Mat image;
	Smoothing(src, image);

	// tim huong sang anh, tinh toan do doc va thuc hien tim diem ung vien bien canh
	GradientComputation_And_Non_MaximumSuppression(image, image);

	// chon cac bien canh dua tren nguong cao, nguong thap
	Hysteresis(image, dst, lowThreshold, highThreshold);

	return 0;
}

// ham dat ten file anh moi de luu anh
string nameImageAterChange(int argc, char** argv) {
	int n = strlen(argv[1]);
	int i = n - 1, j = n;
	while (i >= 0 && argv[1][i] != '\\') {
		i--;
		if (argv[1][i] == '.' && j == n) {
			j = i;
		}
	}
	string newNameImage = "";
	for (i = i + 1; i < j; i++) {
		newNameImage = newNameImage + argv[1][i];
	}
	for (int k = 2; k < argc; k++) {
		newNameImage = newNameImage + argv[k];
	}

	for (int k = j; k < n; k++) {
		newNameImage = newNameImage + argv[1][k];
	}

	return newNameImage;
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		cout << "Chuong trinh tim bien canh" << endl;
		return -1;
	}
	
	Mat image;
	image = imread(argv[1], IMREAD_GRAYSCALE);

	if (!image.data)
	{
		cout << "Khong the mo anh" << std::endl;
		return -1;
	}
	// 20120007_BT01.exe <duongdantaptinanh> (hien thi anh)
	namedWindow("Display window", WINDOW_AUTOSIZE);
	imshow("Display window", image);

	
	if (argc >= 3) {
		string newNameImage = nameImageAterChange(argc, argv);
		Mat image_dst;

		// 20120007_BT01.exe <duongdantaptinanh> sobel (tim bien canh bang thuat Sobel)
		if (strcmp(argv[2], "sobel") == 0) {
			detectBySobel(image, image_dst);
			
			// anh xu li ban Sobel cua openCV
			Mat imageX, imageY, image_openCV;
			// theo hai huong X, Y
			Sobel(image, imageX, image.depth(), 1, 0);
			Sobel(image, imageY, image.depth(), 0, 1);
			// cong 2 dao ham X, Y
			image_openCV = imageX + imageY;

			namedWindow("OpenCV", WINDOW_AUTOSIZE);
			imshow("OpenCV", image_openCV);
		}

		// 20120007_BT01.exe <duongdantaptinanh> prewitt (tim bien canh bang thuat Prewitt)
		else if (strcmp(argv[2], "prewitt") == 0) {
			detectByPrewitt(image, image_dst);
		}

		// 20120007_BT01.exe <duongdantaptinanh> laplace (tim bien canh bang thuat Laplace)
		else if (strcmp(argv[2], "laplace") == 0) {
			detectByLaplace(image, image_dst);

			// anh xu li ban Laplace của OpenCV
			Mat image_openCV;
			Laplacian(image, image_openCV, image.depth());

			namedWindow("OpenCV", WINDOW_AUTOSIZE);
			imshow("OpenCV", image_openCV);
		}

		// 20120007_BT01.exe <duongdantaptinanh> canny (tim bien canh bang thuat Canny)
		else if (strcmp(argv[2], "canny") == 0) {
			// 20120007_BT01.exe <duongdantaptinanh> canny <nguongthap> <nguongcao>(tim bien canh bang thuat Canny)
			if (argc <= 4) {
				detectByCany(image, image_dst);
			}
			else {
				detectByCany(image, image_dst, atoi(argv[3]), atoi(argv[4]));
			}

			Mat image_openCV;
			// lam min anh voi kich thuoc mat na la 5x5
			blur(image, image_openCV, Size(5, 5));

			// anh xu li ban Canny của OpenCV
			if (argc <= 4) {
				Canny(image_openCV, image_openCV, 50, 150);
			}
			else {
				Canny(image_openCV, image_openCV, atoi(argv[3]), atoi(argv[4]));

			}
			namedWindow("OpenCV", WINDOW_AUTOSIZE);
			imshow("OpenCV", image_openCV);
		}

		else {
			cout << "Khong co lenh " << argv[2] << endl;
			waitKey(0);
			return -1;
		}

		namedWindow(argv[2], WINDOW_AUTOSIZE);
		imshow(argv[2], image_dst);

		cout << "Tao anh " << newNameImage << endl;
		imwrite(newNameImage, image_dst);
	}

	waitKey(0);
	return 0;
}

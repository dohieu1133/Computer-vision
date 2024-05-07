#include "opencv2\core.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\highgui\highgui.hpp"
#include <iostream>
#include <string>
#include <cstdlib>
#include <stack>
#include <vector>

using namespace std;
using namespace cv;

#define MAX_FILTER_COLS 5

#define PI 3.14159265358979
#define EXP 2.71828182845904


// ham ma tran kieu "type" qua mat na filter co so cot toi da la 5 va tra ve ma tran kieu double
template <class type>
void Filter(const Mat_<type>& src, Mat& dst, double filter[][MAX_FILTER_COLS], int filterRows, int filterCols) {

	// sao chep ma tran nguon
	Mat image = src.clone();
	// tao ma tran diem anh kich thuoc bang ma tran nguon
	dst = Mat_<double>(src.rows, src.cols);

	double temp;
	int x, y;
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			temp = 0;
			for (int m = 0; m < filterRows; m++) {
				for (int n = 0; n < filterCols; n++) {
					x = i + m - filterRows / 2;
					y = j + n - filterCols / 2;
					// kiem tra x, y co nam trong anh khong va loc diem anh
					if (0 <= x && x < image.rows && 0 <= y && y < image.cols) {
						temp += image.at<type>(x, y) * filter[m][n];
					}
				}
			}
			// luu diem anh vua duoc loc
			dst.at<double>(i, j) = temp;
		}
	}
}

// ham lam min anh, giam nhieu
template <class type>
void Gaussian(const Mat_<type>& src, Mat& dst) {
	// mat na Gaussian
	double GaussianFilter[5][MAX_FILTER_COLS] = { {0.003, 0.013, 0.022, 0.013, 0.003},
												{0.013, 0.059, 0.097, 0.059, 0.013},
												{0.022, 0.097, 0.159, 0.097, 0.022},
												{0.013, 0.059, 0.097, 0.059, 0.013},
												{0.003, 0.013, 0.022, 0.013, 0.003} };

	Filter<type>(src, dst, GaussianFilter, 5, 5);
}

// ham dao ham theo x bang mat na Sobel
template <class type>
void SobelX(const Mat_<type>& src, Mat& dst) {
	// mat na Gaussian
	double SobelXFilter[3][MAX_FILTER_COLS] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };

	Filter<type>(src, dst, SobelXFilter, 3, 3);
}

// ham dao ham theo y bang mat na Sobel
template <class type>
void SobelY(const Mat_<type>& src, Mat& dst) {
	// mat na Gaussian
	double SobelYFilter[3][MAX_FILTER_COLS] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

	Filter<type>(src, dst, SobelYFilter, 3, 3);
}

void redPoint(Mat& img, int i, int j) {
	int x, y;
	for (int m = -1; m <= 1; m++) {
		for (int n = -1; n <= 1; n++) {
			x = i + m;
			y = j + n;
			if (0 <= x && x < img.rows && 0 <= y && y < img.cols) {
				img.at<Vec3b>(x, y)[0] = 0;
				img.at<Vec3b>(x, y)[1] = 0;
				img.at<Vec3b>(x, y)[2] = 255;
			}

			x = i + n;
			y = j + m;
			if (0 <= x && x < img.rows && 0 <= y && y < img.cols) {
				img.at<Vec3b>(x, y)[0] = 0;
				img.at<Vec3b>(x, y)[1] = 0;
				img.at<Vec3b>(x, y)[2] = 255;
			}
		}
	}
}

// ham ve hinh tron ban kinh r tai vi tri (i, j)
void redCircle(Mat& img, int i, int j, int r) {
	if (r == 0) return;

	int x, y1, y2;
	for (int m = -r; m <= r; m++) {
		x = i + m;
		y1 = j + int(sqrt(r * r - m * m));
		y2 = j - int(sqrt(r * r - m * m));
		if (0 <= x && x < img.rows && 0 <= y1 && y1 < img.cols) {
			//redPoint(img, x, y1);

			img.at<Vec3b>(x, y1)[0] = 0;
			img.at<Vec3b>(x, y1)[1] = 0;
			img.at<Vec3b>(x, y1)[2] = 255;
		}

		if (0 <= x && x < img.rows && 0 <= y2 && y2 < img.cols) {
			//redPoint(img, x, y2);
			img.at<Vec3b>(x, y2)[0] = 0;
			img.at<Vec3b>(x, y2)[1] = 0;
			img.at<Vec3b>(x, y2)[2] = 255;
		}
	}
}

Mat keyPoint2Image(const Mat& keyPoint, const Mat& image) {
	Mat result = image.clone();
	for (int i = 0; i < keyPoint.rows; i++) {
		for (int j = 0; j < keyPoint.cols; j++) {
			redCircle(result, i, j, keyPoint.at<double>(i, j));
		}
	}
	return result;
}

// ham lay diem lon nhat cuc bo
template <class type>
Mat localMaxima(const Mat_<type>& image) {

	Mat result = Mat_<double>(image.rows, image.cols);

	int temp;
	int x, y;
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			temp = 0;
			// kiem tra diem hien tai lon nhat cuc bo hay khong
			for (int m = -1; m <= 1; m++) {
				for (int n = -1; n <= 1; n++) {
					x = i + m;
					y = j + n;
					if (0 <= x && x < image.rows && 0 <= y && y < image.cols && image.at<type>(i, j) <= image.at<type>(x, y)) {
						temp++;
					}
				}
			}
			if (temp == 1) {
				result.at<double>(i, j) = 2;
			}
			else {
				result.at<double>(i, j) = 0;
			}
		}
	}
	return result;
}


// ------------------------detectHarrist-------------------------
Mat detectHarrist(Mat img) {
	// doi sang anh xam
	Mat image;
	cvtColor(img, image, COLOR_BGR2GRAY);
	// loc anh qua mat na Gaussian va luu vao Mat_<double> image
	Gaussian<uchar>(image, image);

	// dao ham x, y theo mat na Sobel
	Mat Ix, Iy;
	SobelX<double>(image, Ix);
	SobelY<double>(image, Iy);

	// tinh M
	Mat IxIx, IxIy, IyIy;
	IxIx = Mat_<double>(image.rows, image.cols);
	IxIy = Mat_<double>(image.rows, image.cols);
	IyIy = Mat_<double>(image.rows, image.cols);

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			IxIx.at<double>(i, j) = Ix.at<double>(i, j) * Ix.at<double>(i, j);
			IxIy.at<double>(i, j) = Ix.at<double>(i, j) * Iy.at<double>(i, j);
			IyIy.at<double>(i, j) = Iy.at<double>(i, j) * Iy.at<double>(i, j);
		}
	}

	Gaussian<double>(IxIx, IxIx);
	Gaussian<double>(IxIy, IxIy);
	Gaussian<double>(IyIy, IyIy);

	// tinh lamda
	Mat lamda1, lamda2;
	lamda1 = Mat_<double>(image.rows, image.cols);
	lamda2 = Mat_<double>(image.rows, image.cols);
	double delta, b, c;

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			// (IxIx - lamda) (IyIy - lamda) - IxIy ^ 2 = 0
			// <=> lamda^2 - (IxIx + IyIy) * lamda + IxIx * IyIy - IxIy ^ 2 = 0

			b = -(IxIx.at<double>(i, j) + IyIy.at<double>(i, j));
			c = IxIx.at<double>(i, j) * IyIy.at<double>(i, j) - (IxIy.at<double>(i, j) * IxIy.at<double>(i, j));

			delta = b * b - 4 * c;
			lamda1.at<double>(i, j) = (-b + sqrt(delta)) / 2;
			lamda2.at<double>(i, j) = (-b - sqrt(delta)) / 2;

		}
	}

	// Corner Response Function
	Mat R;
	R = Mat_<double>(image.rows, image.cols);
	double lamda1_ij, lamda2_ij;
	double alpha = 0.05;

	//double maxR = 0;

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			lamda1_ij = lamda1.at<double>(i, j);
			lamda2_ij = lamda2.at<double>(i, j);
			R.at<double>(i, j) = (lamda1_ij * lamda2_ij) - alpha * (lamda1_ij + lamda2_ij) * (lamda1_ij + lamda2_ij);

		}
	}

	Mat result;
	result = Mat_<double>(image.rows, image.cols);
	double T = 1000000;

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {

			if (R.at<double>(i, j) > T) {
				result.at<double>(i, j) = R.at<double>(i, j);
			}
			else {
				result.at<double>(i, j) = 0;
			}
		}
	}

	return localMaxima<double>(result);
}

template <class type>
double value_ij(const Mat_<type>& img, int i, int j, int di, int dj) {
	i += di; j += dj;
	if (0 <= i && i < img.rows && 0 <= j && j < img.cols) {
		return img.at<type>(i, j);
	}
	return 0;
}

// phan phoi gaussian
template <class type>
void Gaussian_sigma(const Mat_<type>& src, Mat& dst, double sigma) {

	int ksize = (int(sigma + 1) * 6) | 1;
	GaussianBlur(src, dst, Size(ksize, ksize), sigma, sigma);
}


// laplacian of gaussian
template <class type>
void LaplacianOfGausian(const Mat_<type>& src, Mat& dst, double sigma) {

	Gaussian_sigma(src, dst, sigma);

	// mat na dao ham cap 2
	double filter[3][MAX_FILTER_COLS] = { {0, 1, 0}, {1, -4, 1}, {0, 1, 0} };
	Filter<double>(dst, dst, filter, 3, 3);

	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {
			dst.at<double>(i, j) *= sigma * sigma;
		}
	}
}

// ------------------------detectBlob-------------------------
Mat detectBlob(Mat img) {
	// doi sang anh xam
	Mat image;
	cvtColor(img, image, COLOR_BGR2GRAY);
	Gaussian<uchar>(image, image);

	double min_sigma = sqrt(2), max_sigma = 30, k = sqrt(2);

	Mat LoG[3];
	int p = 1;

	LaplacianOfGausian<double>(image, LoG[0], min_sigma);

	LaplacianOfGausian<double>(image, LoG[1], min_sigma + k);


	Mat result = Mat_<double>(image.rows, image.cols);
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			result.at<double>(i, j) = 0;
		}
	}

	for (double sigma = min_sigma + k; sigma < max_sigma; sigma += k) {

		LaplacianOfGausian<double>(image, LoG[(p + 1) % 3], sigma + k);

		for (int i = int(sigma * 3 + 2); i < image.rows - int(sigma * 3 + 2); i++) {
			for (int j = int(sigma * 3 + 2); j < image.cols - int(sigma * 3 + 2); j++) {

				if (result.at<double>(i, j) != 0) continue;

				int const_min = 0, const_max = 0, x, y;
				for (int m = -1; m <= 1; m++) {
					for (int n = -1; n <= 1; n++) {
						x = i + m;
						y = j + n;
						if (0 <= x && x < image.rows && 0 <= y && y < image.cols) {
							if (LoG[p].at<double>(i, j) + 0.15 >= LoG[p].at<double>(x, y)) {
								const_min++;
							}
							if (LoG[p].at<double>(i, j) + 0.15 >= LoG[(p + 1) % 3].at<double>(x, y)) {
								const_min++;
							}
							if (LoG[p].at<double>(i, j) + 0.15 >= LoG[(p + 2) % 3].at<double>(x, y)) {
								const_min++;
							}
							if (LoG[p].at<double>(i, j) - 0.15 <= LoG[p].at<double>(x, y)) {
								const_max++;
							}
							if (LoG[p].at<double>(i, j) - 0.15 <= LoG[(p + 1) % 3].at<double>(x, y)) {
								const_max++;
							}
							if (LoG[p].at<double>(i, j) - 0.15 <= LoG[(p + 2) % 3].at<double>(x, y)) {
								const_max++;
							}
						}
					}
				}

				if (const_min == 1 || const_max == 1) {
					result.at<double>(i, j) = sigma * sqrt(2);
				}
				else {
					result.at<double>(i, j) = 0;
				}
			}
		}
		p = (p + 1) % 3;
	}

	return result;
}

// ------------------------detectDoG-------------------------
Mat detectDOG(Mat img) {
	// doi sang anh xam
	Mat image;
	cvtColor(img, image, COLOR_BGR2GRAY);
	Gaussian<uchar>(image, image);

	double min_sigma = sqrt(2), max_sigma = 30, k = sqrt(2);

	Mat G[2];
	Mat DOG[3];
	DOG[0] = Mat_<double>(image.rows, image.cols);
	DOG[1] = Mat_<double>(image.rows, image.cols);
	DOG[2] = Mat_<double>(image.rows, image.cols);

	int p = 1, q = 1;

	Gaussian_sigma<double>(image, G[0], min_sigma);
	Gaussian_sigma<double>(image, G[1], min_sigma + k);
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			DOG[0].at<double>(i, j) = (G[1].at<double>(i, j) - G[0].at<double>(i, j));
		}
	}

	Gaussian_sigma<double>(image, G[0], min_sigma + k + k);
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			DOG[1].at<double>(i, j) = (G[0].at<double>(i, j) - G[1].at<double>(i, j));
		}
	}


	Mat result = Mat_<double>(image.rows, image.cols);
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			result.at<double>(i, j) = 0;
		}
	}

	for (double sigma = min_sigma + k; sigma < max_sigma; sigma += k) {

		Gaussian_sigma<double>(image, G[q], sigma + k + k);
		for (int i = 0; i < image.rows; i++) {
			for (int j = 0; j < image.cols; j++) {
				DOG[(p + 1) % 3].at<double>(i, j) = (G[q].at<double>(i, j) - G[(q + 1) % 2].at<double>(i, j));
			}
		}
		q = (q + 1) % 2;

		for (int i = int(sigma * 3 + 2); i < image.rows - int(sigma * 3 + 2); i++) {
			for (int j = int(sigma * 3 + 2); j < image.cols - int(sigma * 3 + 2); j++) {

				if (result.at<double>(i, j) != 0) continue;

				int const_min = 0, const_max = 0, x, y;
				for (int m = -1; m <= 1; m++) {
					for (int n = -1; n <= 1; n++) {
						x = i + m;
						y = j + n;
						if (0 <= x && x < image.rows && 0 <= y && y < image.cols) {
							if (DOG[p].at<double>(i, j) >= DOG[p].at<double>(x, y)) {
								const_min++;
							}
							if (DOG[p].at<double>(i, j) >= DOG[(p + 1) % 3].at<double>(x, y)) {
								const_min++;
							}
							if (DOG[p].at<double>(i, j) >= DOG[(p + 2) % 3].at<double>(x, y)) {
								const_min++;
							}
							if (DOG[p].at<double>(i, j) <= DOG[p].at<double>(x, y)) {
								const_max++;
							}
							if (DOG[p].at<double>(i, j) <= DOG[(p + 1) % 3].at<double>(x, y)) {
								const_max++;
							}
							if (DOG[p].at<double>(i, j) <= DOG[(p + 2) % 3].at<double>(x, y)) {
								const_max++;
							}
						}
					}
				}
				if (const_min == 1 || const_max == 1) {
					result.at<double>(i, j) = sigma * sqrt(2);
				}
				else {
					result.at<double>(i, j) = 0;
				}
			}
		}
		p = (p + 1) % 3;
	}

	return result;
}
/*
vector<int> SIFT(const Mat& img, int detector) {
	Mat keyPoint;
	if (detector == 1) {
		keyPoint = detectHarrist(img);
	}
	else if (detector == 2) {
		keyPoint = detectBlob(img);
	}
	else if (detector == 2) {
		keyPoint = detectDOG(img);
	}

	// doi sang anh xam
	Mat image;
	cvtColor(img, image, COLOR_BGR2GRAY);
	Gaussian<uchar>(image, image);
	// dao ham x, y theo mat na Sobel
	Mat Ix, Iy;
	SobelX<double>(image, Ix);
	SobelY<double>(image, Iy);
	SobelX<double>(Ix, Ix);
	SobelY<double>(Iy, Iy);
	Mat IxIx, IxIy, IyIy;
	IxIx = Mat_<double>(image.rows, image.cols);
	IxIy = Mat_<double>(image.rows, image.cols);
	IyIy = Mat_<double>(image.rows, image.cols);

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			IxIx.at<double>(i, j) = Ix.at<double>(i, j) * Ix.at<double>(i, j);
			IxIy.at<double>(i, j) = Ix.at<double>(i, j) * Iy.at<double>(i, j);
			IyIy.at<double>(i, j) = Iy.at<double>(i, j) * Iy.at<double>(i, j);
		}
	}

	Gaussian<double>(IxIx, IxIx);
	Gaussian<double>(IxIy, IxIy);
	Gaussian<double>(IyIy, IyIy);

	double temp;

	for (int i = 0; i < keyPoint.rows; i++) {
		for (int j = 0; j < keyPoint.cols; i++) {
			if (keyPoint.at<double>(i, j) != 0) {
				if (sqrt(Ix.at<double>(i, j) * Ix.at<double>(i, j) + Iy.at<double>(i, j) * Iy.at<double>(i, j)) < 0.03) {
					keyPoint.at<double>(i, j) = 0;
				}
				else {
					temp = (IxIx.at<double>(i, j) + IyIy.at<double>(i, j));
					temp *= temp;
					temp /= IxIx.at<double>(i, j) * IyIy.at<double>(i, j) + IxIy.at<double>(i, j) * IxIy.at<double>(i, j);
					if (temp <= (11 * 11 / 10)) {
						keyPoint.at<double>(i, j) = 0;
					}
				}
			}
		}
	}
}
*/

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
		cout << "Chuong trinh tim dac trung anh" << endl;
		return -1;
	}

	Mat image, keyPoint;
	image = imread(argv[1], IMREAD_COLOR);
	if (!image.data)
	{
		cout << "Khong the mo anh" << std::endl;
		return -1;
	}
	// 20120007_Lab03.exe <duongdantaptinanh> (hien thi anh)
	namedWindow("Display window", WINDOW_AUTOSIZE);
	imshow("Display window", image);


	if (argc >= 3) {
		string newNameImage = nameImageAterChange(argc, argv);
		Mat image1;

		// 20120007_Lab03.exe <duongdantaptinanh> harrist (tim dac chung goc)
		if (strcmp(argv[2], "harrist") == 0) {
			keyPoint = detectHarrist(image);
			namedWindow("Harrist", WINDOW_AUTOSIZE);
			image1 = keyPoint2Image(keyPoint, image);
			imshow("Harrist", image1);
		}

		// 20120007_Lab03.exe <duongdantaptinanh> blob (tim dac chung dom sang bang laplacian of gaussian)
		else if (strcmp(argv[2], "blob") == 0) {
			keyPoint = detectBlob(image);
			namedWindow("blob", WINDOW_AUTOSIZE);
			image1 = keyPoint2Image(keyPoint, image);
			imshow("blob", image1);
		}

		// 20120007_Lab03.exe <duongdantaptinanh> dog (tim dac chung dom sang bang DOG)
		else if (strcmp(argv[2], "dog") == 0) {
			keyPoint = detectDOG(image);
			namedWindow("dog", WINDOW_AUTOSIZE);
			image1 = keyPoint2Image(keyPoint, image);
			imshow("dog", image1);
		}

		else {
			cout << "Khong co lenh " << argv[2] << endl;
			waitKey(0);
			return -1;
		}

		cout << "Tao anh " << newNameImage << endl;
		imwrite(newNameImage, image1);
	}
	waitKey(0);
	return 0;

}
//// SFM.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
////
//
#include "opencv.hpp"
#include <opencv2\xfeatures2d\nonfree.hpp>

#include <pcl/visualization/cloud_viewer.h>
#include<pcl/point_types.h>  
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>

#include <iostream> 
#include <fstream>
#include <string>

using namespace cv;
using namespace std;


#define BOARD_WIDTH 22   //实际测量得到的标定板上每个棋盘格width的大小
#define BOARD_HEIGHT 23  //实际测量得到的标定板上每个棋盘格height的大小


void image_format()
{
	//图片格式转化
	cout << "开始执行图片格式转化" << endl;
	ifstream fin_for_jpg("image_contents_for_jpg.txt"); //标定所用图像文件的jpg文件的路径，需要在该文件下根据需要修改image_contents_for_jpg.txt
	string filename_jpg;
	int image_count_jpg = 0;
	while (getline(fin_for_jpg, filename_jpg))
	{
		image_count_jpg++;
		Mat M = imread(filename_jpg);   // 读入图片 
		if (M.empty())     // 判断文件是否正常打开  
		{
			fprintf(stderr, "Can not load image %s\n", filename_jpg);
			waitKey(6000);  // 等待6000 ms后窗口自动关闭   
			exit(1);
		}
		//需要的话打开可以查看jpg图片信息
		//cout<<M.channels()<<endl;
		//cout<<M.rows<<endl;
		//cout<<M.cols<<endl;
		//cout<<M.type();

		//imshow("image",M);  // 显示图片 
		char buffer[999];
		char tail1[] = ".bmp";
		sprintf(buffer, "%s%d%s", "picture", image_count_jpg, tail1);
		imwrite(buffer, M); // 存为bmp格式图片
	}
	cout << "图片格式转化完成" << endl;
	return;
}

void calibrate(Mat &cameraMatrix)
{	
	//image_format();
	//正式开始
	ifstream fin("image_contents.txt"); //标定所用图像文件的bmp格式的路径，需要在该文件下根据需要修改image_contents.txt
	//读取每一幅图像，从中提取出角点，然后对角点进行亚像素精确化	
	//cout<<"开始提取角点………………"<<endl;
	int image_count = 0;  //图像数量计数器
	Size image_size;  //图像的尺寸
	Size board_size = Size(7, 9);    // 标定板上每行、列的角点数，width=7,height=9
	vector<Point2f> image_points_temp;  //暂存每幅图像上检测到的角点的容器
	vector<vector<Point2f>> image_points; //保存检测到的所有角点的容器
	string filename;
	int count = -1;//用于存储角点个数。
	cout << "标定图片开始处理" << endl;
	while (getline(fin, filename))
	{
		image_count++;
		// 用于观察检验输出
		//cout << "第" << image_count << "张图片开始处理" << endl;
		Mat image = imread(filename);
		if (image.empty()) {
			// 判断文件是否正常打开  
			cout << "读取此文件出错，请检查图片是否存在" << endl;
			system("pause");
			continue;
		}
		image_size.width = image.cols;
		image_size.height = image.rows;
		//cout << "image_width_" << image_count << "= " << image_size.width << endl;
		//cout << "image_height_" << image_count << "= " << image_size.height << endl;
		//提取角点
		if (0 == findChessboardCorners(image, board_size, image_points_temp))
		{
			cout << "无法找到角点，请重试此图片!" << endl; //找不到角点
			system("pause");
			exit(1);
		}
		else
		{
			Mat image_gray;
			cvtColor(image, image_gray, CV_RGB2GRAY);//将彩色图像转化为灰度图像
			//亚像素精确化
			find4QuadCornerSubpix(image_gray, image_points_temp, Size(5, 5)); //对粗提取的角点进行精确化
			//Size(5,5)为角点搜索窗口的尺寸
			image_points.push_back(image_points_temp);  //保存亚像素角点
		}
	}
	fin.close();
	fin.clear();
	cout << "角点提取完成，即将进行相机标定" << endl;

	//相机标定
	//棋盘三维信息
	Size square_size = Size(BOARD_WIDTH, BOARD_HEIGHT);  //定义实际测量得到的标定板上每个棋盘格的大小(width,height)
	vector<vector<Point3f>> object_points; //用以保存标定板上角点的三维坐标
	/*内外参数*/
	//Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); //相机内参数矩阵
	Mat distCoeffs = Mat(1, 5, CV_32FC1, Scalar::all(0)); //相机的5个畸变系数：k1,k2,p1,p2,k3
	vector<Mat> tvecsMat;  //每幅图像的旋转向量
	vector<Mat> rvecsMat; //每幅图像的平移向量

	vector<int> point_counts;  // 定义一个容器储存每幅图像中角点的数量
	//初始化每幅图像中的完整的标定板的角点数量
	for (int i = 0; i < image_count; i++) {
		point_counts.push_back(board_size.width*board_size.height);//将每幅图像中的角点数量依次存入容器
	}

	//初始化标定板上角点的三维坐标
	for (int t = 0; t < image_count; t++) {
		//循环图片
		vector<Point3f> PointSet_temp;//用以暂时保存每一张标定板上角点的三维坐标
		for (int i = 0; i < board_size.height; i++) {
			//循环行
			for (int j = 0; j < board_size.width; j++) {
				//循环列
				Point3f realPoint;//真实点，含有x,y,z三个坐标
				realPoint.x = i * square_size.width;//行坐标
				realPoint.y = j * square_size.height;//列坐标
				realPoint.z = 0;//假设标定板放在世界坐标系中z=0的平面上
				PointSet_temp.push_back(realPoint);//逐个存入一张图中所有的角点
			}
		}
		object_points.push_back(PointSet_temp);//将一张图中所有的角点的容器中存入object_points
	}

	//开始进行相机标定
	calibrateCamera(object_points, image_points, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, CV_CALIB_RATIONAL_MODEL);
	cout << "相机参数获取完成" << endl;
}

//提取所有图像的特征
void extract_features(vector<string>& image_names,vector<vector<KeyPoint>>& key_points_for_all,vector<Mat>& descriptor_for_all,vector<vector<Vec3b>>& colors_for_all)
{
	key_points_for_all.clear();
	descriptor_for_all.clear();
	Mat image;

	//读取图像，获取图像特征点，并保存
	Ptr<Feature2D> sift = xfeatures2d::SIFT::create(0, 3, 0.04, 10);
	for (auto it = image_names.begin(); it != image_names.end(); ++it)
	{
		image = imread(*it);
		if (image.empty()) continue;

		cout << "提取" << *it << "的图像特征" << endl;

		vector<KeyPoint> key_points;
		Mat descriptor;
		//偶尔出现内存分配失败的错误
		sift->detectAndCompute(image, noArray(), key_points, descriptor);

		//特征点过少，则排除该图像
		if (key_points.size() <= 10) continue;

		key_points_for_all.push_back(key_points);
		descriptor_for_all.push_back(descriptor);

		vector<Vec3b> colors(key_points.size());
		for (int i = 0; i < key_points.size(); ++i)
		{
			Point2f& p = key_points[i].pt;
			colors[i] = image.at<Vec3b>(p.y, p.x);
		}
		colors_for_all.push_back(colors);
	}
}

void match_features1(Mat& query, Mat& train, vector<DMatch>& matches)
{
	vector<vector<DMatch>> knn_matches;
	BFMatcher matcher(NORM_L2);
	matcher.knnMatch(query, train, knn_matches, 2);

	//获取满足Ratio Test的最小匹配的距离
	float min_dist = FLT_MAX;
	for (int r = 0; r < knn_matches.size(); ++r)
	{
		//Ratio Test
		if (knn_matches[r][0].distance > 0.6*knn_matches[r][1].distance)
			continue;

		float dist = knn_matches[r][0].distance;
		if (dist < min_dist) min_dist = dist;
	}

	matches.clear();
	for (size_t r = 0; r < knn_matches.size(); ++r)
	{
		//排除不满足Ratio Test的点和匹配距离过大的点
		if (
			knn_matches[r][0].distance > 0.6*knn_matches[r][1].distance ||
			knn_matches[r][0].distance > 5 * max(min_dist, 10.0f)
			)
			continue;

		//保存匹配点
		matches.push_back(knn_matches[r][0]);
	}
}

void match_features2(vector<Mat>& descriptor_for_all, vector<vector<DMatch>>& matches_for_all)
{
	matches_for_all.clear();
	// n个图像，两两顺次有 n-1 对匹配
	// 1与2匹配，2与3匹配，3与4匹配，以此类推
	for (int i = 0; i < descriptor_for_all.size() - 1; ++i)
	{
		cout << "第 " << i + 1 << " 与第 " << i + 2 << "张图片进行匹配" << endl;
		vector<DMatch> matches;
		match_features1(descriptor_for_all[i], descriptor_for_all[i + 1], matches);
		matches_for_all.push_back(matches);
	}
}


bool find_transform(Mat& K, vector<Point2f>& p1, vector<Point2f>& p2, Mat& R, Mat& T, Mat& mask)
{
	//根据内参矩阵获取相机的焦距和光心坐标（主点坐标）
	double focal_length = 0.5*(K.at<double>(0) + K.at<double>(4));
	Point2d principle_point(K.at<double>(2), K.at<double>(5));

	//根据匹配点求取本征矩阵，使用RANSAC，进一步排除失配点
	Mat E = findEssentialMat(p1, p2, focal_length, principle_point, RANSAC, 0.999, 1.0, mask);
	if (E.empty()) {
		cout << "E is empty" << endl;
		return false;
	}

	//可行点的数量
	double feasible_count = countNonZero(mask);
	cout << "匹配得到的可行点的数量为："<<(int)feasible_count << " -in- " << p1.size() << endl;

	//分解本征矩阵，获取相对变换
	int pass_count = recoverPose(E, p1, p2, R, T, focal_length, principle_point, mask);

	//对于RANSAC而言，outlier数量大于50%时，结果是不可靠的
	if (feasible_count <= 15 || (feasible_count / p1.size()) < 0.6) {
		cout << "outlier数量大于50%，结果是不可靠" << endl;
		return false;
	}
		

	//同时位于两个相机前方的点的数量要足够大
	if (((double)pass_count) / feasible_count < 0.7) {
		cout << "同时位于两个相机前方的点的数量不够大" << endl;
		return false;
	}
		

	cout << "find_transform Succeed" << endl;
	return true;
}

//获得匹配点
void get_matched_points(vector<KeyPoint>& p1,vector<KeyPoint>& p2,vector<DMatch> matches,vector<Point2f>& out_p1,vector<Point2f>& out_p2)
{
	out_p1.clear();
	out_p2.clear();
	for (int i = 0; i < matches.size(); ++i)
	{
		out_p1.push_back(p1[matches[i].queryIdx].pt);
		out_p2.push_back(p2[matches[i].trainIdx].pt);
	}
}

//获得匹配的colors
void get_matched_colors(vector<Vec3b>& c1,vector<Vec3b>& c2,vector<DMatch> matches,vector<Vec3b>& out_c1,vector<Vec3b>& out_c2)
{
	out_c1.clear();
	out_c2.clear();
	for (int i = 0; i < matches.size(); ++i)
	{
		out_c1.push_back(c1[matches[i].queryIdx]);
		out_c2.push_back(c2[matches[i].trainIdx]);
	}
}

//三维重建
void reconstruct(Mat& K, Mat& R1, Mat& T1, Mat& R2, Mat& T2, vector<Point2f>& p1, vector<Point2f>& p2, vector<Point3f>& structure)
{
	//两个相机的投影矩阵[R T]，triangulatePoints只支持float型
	Mat proj1(3, 4, CV_32FC1);
	Mat proj2(3, 4, CV_32FC1);

	R1.convertTo(proj1(Range(0, 3), Range(0, 3)), CV_32FC1);
	T1.convertTo(proj1.col(3), CV_32FC1);

	R2.convertTo(proj2(Range(0, 3), Range(0, 3)), CV_32FC1);
	T2.convertTo(proj2.col(3), CV_32FC1);

	Mat fK;
	K.convertTo(fK, CV_32FC1);
	proj1 = fK * proj1;
	proj2 = fK * proj2;

	//三角重建
	Mat s;
	triangulatePoints(proj1, proj2, p1, p2, s);

	structure.clear();
	structure.reserve(s.cols);
	for (int i = 0; i < s.cols; ++i)
	{
		Mat_<float> col = s.col(i);
		col /= col(3);	//齐次坐标，需要除以最后一个元素才是真正的坐标值
		structure.push_back(Point3f(col(0), col(1), col(2)));
	}
}

void maskout_points(vector<Point2f>& p1, Mat& mask)
{
	vector<Point2f> p1_copy = p1;
	p1.clear();

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
			p1.push_back(p1_copy[i]);
	}
}

void maskout_colors(vector<Vec3b>& p1, Mat& mask)
{
	vector<Vec3b> p1_copy = p1;
	p1.clear();

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
			p1.push_back(p1_copy[i]);
	}
}

//保存strucure
void save_structure(string file_name, vector<Mat>& rotations, vector<Mat>& motions, vector<Point3f>& structure, vector<Vec3b>& colors)
{
	int n = (int)rotations.size();

	ofstream fout(file_name);  //保存标定结果的文件
	fout << "Camera Count" << n << endl;
	fout << "Point Count" << (int)structure.size() << endl;
	fout << "相机位姿（相机外参）：Rotations + Motions" << endl;
	
	for (size_t i = 0; i < n; ++i)
	{
		fout << "Rotations(旋转）" << endl;
		fout << rotations[i] << endl;
		fout << "Motions(平移）" << endl;
		fout << motions[i] << endl;
	}

	fout << "Points" << "[";
	for (size_t i = 0; i < structure.size(); ++i)
	{
		fout << structure[i] << endl;
	}
	fout << "]";

	fout << "Colors" << "[";
	for (size_t i = 0; i < colors.size(); ++i)
	{
		fout << colors[i] << endl;
	}
	fout << "]";
}

//获得真实点和图像上的点
void get_objpoints_and_imgpoints(vector<DMatch>& matches,vector<int>& struct_indices,vector<Point3f>& structure,vector<KeyPoint>& key_points,vector<Point3f>& object_points,vector<Point2f>& image_points)
{
	object_points.clear();
	image_points.clear();

	for (int i = 0; i < matches.size(); ++i)
	{
		int query_idx = matches[i].queryIdx;
		int train_idx = matches[i].trainIdx;

		int struct_idx = struct_indices[query_idx];
		if (struct_idx < 0) continue;

		object_points.push_back(structure[struct_idx]);
		image_points.push_back(key_points[train_idx].pt);
	}
}

//将新的重建结果与之前的融合
void fusion_structure(vector<DMatch>& matches,vector<int>& struct_indices,vector<int>& next_struct_indices,vector<Point3f>& structure,vector<Point3f>& next_structure,vector<Vec3b>& colors,vector<Vec3b>& next_colors)
{
	for (int i = 0; i < matches.size(); ++i)
	{
		int query_idx = matches[i].queryIdx;
		int train_idx = matches[i].trainIdx;

		int struct_idx = struct_indices[query_idx];
		if (struct_idx >= 0) //若该点在空间中已经存在，则这对匹配点对应的空间点应该是同一个，索引要相同
		{
			next_struct_indices[train_idx] = struct_idx;
			continue;
		}

		//若该点在空间中已经存在，将该点加入到结构中，且这对匹配点的空间点索引都为新加入的点的索引
		structure.push_back(next_structure[i]);
		colors.push_back(next_colors[i]);
		struct_indices[query_idx] = next_struct_indices[train_idx] = structure.size() - 1;
	}
}

//初始化三维点云结构
void init_structure(Mat K,vector<vector<KeyPoint>>& key_points_for_all,vector<vector<Vec3b>>& colors_for_all,vector<vector<DMatch>>& matches_for_all,vector<Point3f>& structure,vector<vector<int>>& correspond_struct_idx,
	vector<Vec3b>& colors,vector<Mat>& rotations,vector<Mat>& motions)
{
	//计算头两幅图像之间的变换矩阵
	vector<Point2f> p1, p2;
	vector<Vec3b> c2;
	
	//旋转矩阵和平移向量
	Mat R, T;	
	
	//mask中大于零的点代表匹配点，等于零代表失配点
	Mat mask;

	get_matched_points(key_points_for_all[0], key_points_for_all[1], matches_for_all[0], p1, p2);
	get_matched_colors(colors_for_all[0], colors_for_all[1], matches_for_all[0], colors, c2);

	cout << "统计可行点的数量" << endl;
	find_transform(K, p1, p2, R, T, mask);

	//对头两幅图像进行三维重建
	maskout_points(p1, mask);
	maskout_points(p2, mask);
	maskout_colors(colors, mask);

	Mat R0 = Mat::eye(3, 3, CV_64FC1);
	Mat T0 = Mat::zeros(3, 1, CV_64FC1);
	reconstruct(K, R0, T0, R, T, p1, p2, structure);
	
	//保存变换矩阵
	rotations = { R0, R };
	motions = { T0, T };

	//将correspond_struct_idx的大小初始化为与key_points_for_all完全一致
	correspond_struct_idx.clear();
	correspond_struct_idx.resize(key_points_for_all.size());
	for (int i = 0; i < key_points_for_all.size(); ++i)
	{
		correspond_struct_idx[i].resize(key_points_for_all[i].size(), -1);
	}

	//填写头两幅图像的结构索引
	int idx = 0;
	vector<DMatch>& matches = matches_for_all[0];
	for (int i = 0; i < matches.size(); ++i)
	{
		if (mask.at<uchar>(i) == 0)
			continue;

		correspond_struct_idx[0][matches[i].queryIdx] = idx;
		correspond_struct_idx[1][matches[i].trainIdx] = idx;
		++idx;
	}
}

//获取SFM待处理的图片
void get_file_names(string &dir_name, vector<string> & img_names)
{
	//SFM待处理的图片名vector
	string filename;
	ifstream fin(dir_name); //标定所用图像文件的bmp格式的路径，需要在该文件下根据需要修改image_contents.txt
	/*string img1 = "hongtian_huang_2.bmp";
	string img2 = "hongtian_huang_3.bmp";
	string img3 = "hongtian_huang_4.bmp";*/
		
	int count = 1;
	while (getline(fin, filename))
	{
		// 用于观察检验输出
		cout << "尝试打开第" << count << "张待重建图片检查可用性" << endl;
		Mat image = imread(filename);
		if (image.empty()) {
			// 判断文件是否正常打开  
			cout << "读取此文件出错，请检查图片是否存在" << endl;
			system("pause");
			continue;
		}
		img_names.push_back(filename);
		count++;
	}
	fin.close();
}

void main()
{
	image_format();
	//Mat K;

	//第二题本征矩阵
	/*Mat K(Matx33d(
		2899.3, 0, 783.9644,
		0, 2890, 666.4497,
		0, 0, 1));*/


	//第三题第一问本征矩阵
	/*Mat K(Matx33d(
		3437.84, 0, 3127.19,
		0, 3435.95, 2066.98,
		0, 0, 1));*/
	
	//第三题第二问本征矩阵
	Mat K(Matx33d(
		3408.35, 0, 3114.7,
		0, 3408.8, 2070.92,
		0, 0, 1));

	//calibrate(K);
	
	//存储待重建的图片名称
	vector<string> img_names;
	//待重建的图片名称列表的txt名称
	string images = "image_SFM_contents.txt";
	//读取图片
	get_file_names(images, img_names);

	//特征点
	vector<vector<KeyPoint>> key_points_for_all;
	//描述算子
	vector<Mat> descriptor_for_all;
	vector<vector<Vec3b>> colors_for_all;
	vector<vector<DMatch>> matches_for_all;
	
	//提取所有图像的特征
	extract_features(img_names, key_points_for_all, descriptor_for_all, colors_for_all);
	//对所有图像进行顺次的特征匹配
	match_features2(descriptor_for_all, matches_for_all);

	//保存最终匹配到的三维点
	vector<Point3f> structure;

	//保存第i副图像中第j个特征点对应的structure中点的索引
	vector<vector<int>> correspond_struct_idx; 
	vector<Vec3b> colors;
	
	//变化矩阵
	vector<Mat> rotations;
	vector<Mat> motions;

	//初始化结构（三维点云）
	init_structure(K,key_points_for_all,colors_for_all,matches_for_all,structure,correspond_struct_idx,colors,rotations,motions);

	//增量方式重建剩余的图像
	for (int i = 0; i < matches_for_all.size(); ++i)
	{
		//真实点vector
		vector<Point3f> object_points;
		//图像中的点vector
		vector<Point2f> image_points;
		//旋转向量
		Mat r;
		//旋转矩阵
		Mat R, T;
		//Mat mask;

		//获取第i幅图像中匹配点对应的三维点，以及在第i+1幅图像中对应的像素点
		get_objpoints_and_imgpoints(matches_for_all[i],correspond_struct_idx[i],structure,key_points_for_all[i + 1],object_points,image_points);

		//求解变换矩阵
		solvePnPRansac(object_points, image_points, K, noArray(), r, T);
		//将旋转向量转换为旋转矩阵
		Rodrigues(r, R);
		
		//保存变换矩阵
		rotations.push_back(R);
		motions.push_back(T);

		vector<Point2f> p1, p2;
		vector<Vec3b> c1, c2;

		//特征匹配
		get_matched_points(key_points_for_all[i], key_points_for_all[i + 1], matches_for_all[i], p1, p2);
		get_matched_colors(colors_for_all[i], colors_for_all[i + 1], matches_for_all[i], c1, c2);

		//根据之前求得的R，T进行三维重建
		vector<Point3f> next_structure;
		reconstruct(K, rotations[i], motions[i], R, T, p1, p2, next_structure);

		//将新的重建结果与之前的融合
		fusion_structure(matches_for_all[i],correspond_struct_idx[i],correspond_struct_idx[i + 1],structure,next_structure,colors,c1);


		Mat img_matches;
		Mat img_with_keypoints;
		Mat img_with_keypointss;
		Mat image_1 = imread(img_names[i]);
		Mat image_2 = imread(img_names[i+1]);
		drawMatches(image_1, key_points_for_all[i], image_2, key_points_for_all[i+1], matches_for_all[i], img_matches);
		
		drawKeypoints(image_1, key_points_for_all[i], img_with_keypoints, Scalar(0, 255, 0));
		char buffer[999];
		char tail[] = ".bmp";
		sprintf(buffer, "%s%d%s", "matches", i+1, tail);
		imwrite(buffer, img_matches); // 存为bmp格式图片

		char buffers[999];
		sprintf(buffers, "%s%d%s", "img_with_keypoints", i + 1, tail);
		imwrite(buffers, img_with_keypoints); // 存为bmp格式图片

		if(i == matches_for_all.size()-1){
			drawKeypoints(image_2, key_points_for_all[i + 1], img_with_keypointss, Scalar(0, 255, 0));

			char bufferss[999];
			sprintf(bufferss, "%s%d%s", "img_with_keypoints", i + 2, tail);
			imwrite(bufferss, img_with_keypointss); // 存为bmp格式图片
		}
	}


	//输出可视化点云查看
	pcl::PointCloud<pcl::PointXYZ> pointcloud;
	int ptCount_total= structure.size();
	pointcloud.width = ptCount_total;
	pointcloud.height = 1;
	pointcloud.is_dense = false;
	pointcloud.points.resize(pointcloud.width * pointcloud.height);
	
	//定义已经存储过的点数
	int base=0;
	
	for (int i = 0; i < ptCount_total; i++) {
		pointcloud[i].x = structure[i].x;
		pointcloud[i].y = structure[i].y;
		pointcloud[i].z = structure[i].z;
		}

	pcl::io::savePCDFileASCII("pointcloud_pdb.pdb", pointcloud);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZ>); // 创建点云（指针） 
	pcl::io::loadPCDFile("pointcloud_pdb.pdb", *cloud1);
	pcl::visualization::PCLVisualizer viewer;
	pcl::io::savePLYFile("pointcloud_ply.ply", *cloud1);
	viewer.addPointCloud(cloud1, "cloud");
	cout << "点云结果如图所示" << endl;
	viewer.spin();
	//system("pause");

	//保存structure
	save_structure("structure.txt", rotations, motions, structure, colors);
}


// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件

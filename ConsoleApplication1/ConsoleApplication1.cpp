// ConsoleApplication1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracking.hpp>
using namespace std;
using namespace cv;

const int frameskip = 2;

class tracked
{
public:
	tracked(int id)
	{
		this->id = id;
		missingfor = 0;
		startx = 0;
		starty = 0;
		isstillhere = true;
		tracker = TrackerKCF::create();
	}
	int id;
	Ptr<Tracker> tracker;
	Ptr<Rect> rect;
	int startx;
	int starty;
	int missingfor;
	bool isstillhere;
	long since;
	void incrementcounter()
	{
		this->missingfor++;
	}
	vector< Mat> features;
};

int main()
{
	Ptr<FaceDetectorYN> faceDetector = FaceDetectorYN::create("./yunet.onnx", "", cv::Size(640, 480));
	Ptr<FaceRecognizerSF> faceRecognizer = cv::FaceRecognizerSF::create("./face_recognizer_fast.onnx", "");

	//Ptr<FaceRecognizer> faceRecognizer = FaceRecognizer::create("./face_recognizer_fast.onnx", "");

	int lastid = 0;
	long framenr = 0;
	auto hog = cv::HOGDescriptor();
	hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
	auto faceCascadePath = "./haarcascade_frontalface.xml";
	auto faceCascade = cv::CascadeClassifier();
	faceCascade.load(faceCascadePath);

	std::vector<Rect> faces;


	cv::VideoCapture cap;
	if (!cap.open("rtsp://admin:Omega99Omega99!@192.168.15.98/media/video2")) {
		std::cout << "Unable to open video capture\n";
		return -1;
	}
	Rect2d roi;
	vector<Ptr<tracked>> trackedbodies;

	while (true) {
		cv::Mat frame;


		auto ret = cap.grab();
		cap >> frame;
		framenr++;
		if (framenr % frameskip == 0)
		{

			cv::resize(frame, frame, cv::Size(640, 480));

			if (frame.empty()) {
				break; // End of video stream
			}

			Mat gray, smallImg(cvRound(frame.rows), cvRound(frame.cols), CV_8UC1);
			cv::cvtColor(frame, gray, cv::COLOR_RGB2GRAY);
			Mat faces;
			faceDetector->detect(frame, faces);
			// detect people in the image
			// returns the bounding boxes for the detected objects

			vector<Rect> boxes;
			//hog.detectMultiScale(gray, boxes, 0, Size(4, 4), Size(8, 8));
			//faceCascade.detectMultiScale(frame, boxes);

			//for (int i = 0; i < faces.rows; i++)
			//{
			//	Mat aligned_face;
			//	faceRecognizer->alignCrop(frame, faces.row(i), aligned_face);

			//	float src_point[5][2];
			//	for (int row = 0; row < 5; ++row)
			//	{
			//		for (int col = 0; col < 2; ++col)
			//		{
			//			src_point[row][col] = faces.row(i).at<float>(0, row * 2 + col + 4);
			//		}
			//	}

			//	Mat feature;
			//	faceRecognizer->feature(aligned_face, feature);
			//	feature = feature.clone();

			//	bool found = false;
			//	for (Ptr<tracked> t : trackedbodies)
			//	{
			//		if (t->isstillhere)
			//		for (int ii = 0; ii < t->features.size(); ii++)
			//		{
			//			double cos_score = faceRecognizer->match(feature, t->features[ii], FaceRecognizerSF::DisType::FR_COSINE);
			//			// Calculating the discrepancy between two face features by using normL2 distance.
			//			double L2_score = faceRecognizer->match(feature, t->features[ii], FaceRecognizerSF::DisType::FR_NORM_L2);
			//			if (cos_score >= 0.363 && L2_score <= 1.128)
			//			{
			//				ii == t->features.size();
			//				found = true;
			//				t->isstillhere = framenr;
			//				cv::imshow(to_string(t->id), aligned_face);
			//			}
			//		}
			//	}
			//	if (!found)
			//	{
			//		Ptr<tracked> tr = new tracked(lastid);
			//		lastid++;
			//		tr->startx = src_point[0][0];
			//		tr->starty = src_point[0][1];
			//		//tr->rect = new cv::Rect(r);
			//		//tr->tracker->init(frame, *tr->rect);
			//		tr->missingfor = 0;
			//		tr->since = framenr;
			//		tr->features.push_back(feature);
			//		trackedbodies.push_back(tr);
			//		//cv::rectangle(frame, r, cv::Scalar(0, 255, 0));


			//	}

			//}

			//if (framenr % 100 == 0)
			//	for (Ptr<tracked> t : trackedbodies)
			//	{
			//		for (Ptr<tracked> t2 : trackedbodies)
			//		{
			//			if (t->id != t2->id) {
			//				for (int ii = 0; ii < t->features.size(); ii++) {
			//					for (int ii2 = 0; ii2 < t2->features.size(); ii2++) {
			//					double cos_score = faceRecognizer->match(t2->features[ii2], t->features[ii], FaceRecognizerSF::DisType::FR_COSINE);
			//					// Calculating the discrepancy between two face features by using normL2 distance.
			//					double L2_score = faceRecognizer->match(t2->features[ii2], t->features[ii], FaceRecognizerSF::DisType::FR_NORM_L2);
			//					if (cos_score >= 0.363 && L2_score <= 1.128)
			//					{
			//						t2->isstillhere = false;
			//					}
			//				}
			//			}
			//		}
			//	}




			try
			{
				for (Ptr<tracked> t : trackedbodies)
				{
					try {
						bool ishere = t->tracker->update(frame, *t->rect);
						t->isstillhere = ishere;
					}
					catch (const std::exception&)
					{
						t->isstillhere = false;
					}

					if (t->isstillhere)
					{
						cv::rectangle(frame, *t->rect, cv::Scalar(255, 0, 0));
					}
					else
					{
						t->missingfor++;
					}
				}

				for (int i = 0; i < faces.rows; i++) {
					bool found = false;
			
					Rect r;

					r.x = faces.row(i).at<float>(0, 0);
					r.y = faces.row(i).at<float>(0, 1);
					r.height = faces.row(i).at<float>(0, 3);
					r.width = faces.row(i).at<float>(0, 2);

					for (Ptr<tracked> t : trackedbodies)
					{
						if (t->isstillhere)
						{
							auto intersection = *t->rect & r;
							if (intersection.area() > 0) {
								found = true;
							}
						}
					}
					if (!found)
					{
						Ptr<tracked> tr = new tracked(lastid);
						lastid++;
						tr->startx = r.x;
						tr->starty = r.y;
						tr->rect = new cv::Rect(r);
						tr->tracker->init(frame, *tr->rect);
						tr->missingfor = 0;
						tr->since = framenr;
						trackedbodies.push_back(tr);
						cv::rectangle(frame, r, cv::Scalar(0, 255, 0));
					}
				}

				for (vector<Ptr<tracked>>::const_iterator it = trackedbodies.begin(); it != trackedbodies.end(); ) {
					if ((*it)->missingfor > 20) {
						it = trackedbodies.erase(it);
					}
					else {
						++it;
					}
				}

			}
			catch (const std::exception&)
			{

			}


			//cv::resize(frame, frame, cv::Size(640, 480));

			// Do other stuff here with frame

			cv::imshow("frame", frame);

			if (cv::waitKey(10) == 27) {
				break; // stop capturing by pressing ESC
			}
		}
	}

	return 0;
}
// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file

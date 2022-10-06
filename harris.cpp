#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

Mat src, gradX, gradY, Dx, Dy, H;
double x, y;

double harris(Mat& Gx, Mat& Gy, int x, int y, int w, Mat& H)
{
    // Point d'intéret = x,y
    // On prend la somme Dx + Dy
    // Ensuite on initialise H en fonction de 2w+1 (2*2)
    // Prendre le point centrale de la matrice Dx Dy (width /2 +  length / 2)
    // On remplis la matrice H par rapport à la largeur 2w+1 en évitant
    Mat result;
    H = Mat(2, 2, CV_64FC1);
    //if (x - H.rows / 2 < 0 || x + H.rows / 2 || y - )
    add(Gx, Gy, result);
    // On ajoute un padding
    copyMakeBorder(result, result, 2*w+1, 2*w+1, 2*w+1, 2*w+1, BORDER_CONSTANT, 0);

    double xy = 0;
    double X = 0;
    double Y = 0;
    for (int i = -w; i <= w; ++i) {
        for (int j = -w; j <= w; ++j) {
            xy +=  Gx.at<double>(y+j, x+i) * Gy.at<double>(y+j, x+i);
            X += pow(Gx.at<double>(y+j, x+i), 2); // Gx = Ix
            Y += pow(Gy.at<double>( y+j, x+i), 2); // Gy = Iy
            //H.at<double>(x+i, x+j+1) = Gx.at<double>(x, y) * Gy.at<double>(x, y);
            //H.at<double>(x+i+1, x+j) = Gx.at<double>(x, y) * Gy.at<double>(x, y);
        }
    }

    // Valeur propre = eigen
    // Produit des valaurs propres - 

    H.at<double>(0, 0) = X;
    H.at<double>(0, 1) = xy;
    H.at<double>(1, 0) = xy;
    H.at<double>(1, 1) = Y;
    double k = 0.2;
  return determinant(H) - k * pow(trace(H)[0], 2);  // doit en fait retourner det(H)-0.15 tr^2(H)
}


void mouse_callback(int event, int x, int y, int flags, void* unused)
{
if (event == EVENT_LBUTTONDOWN)
  {
	cout << "Vous avez cliqué sur le point (" << x << ", " << y << ")\n";
    cout << "Harris : " << harris(Dx, Dy, x, y, 1, H) << endl;
    cout << H << endl;
    Mat vp;
    Mat VectVp;
    eigen(H, vp, VectVp);
    //cout << "Valeur propres : " << vp << endl;
    cout << VectVp << endl;
	// A vous de compléter
  }
}

/*
 * Calcule et stock dans K les valeurs de la première dérivée partielle par rapport à x d'une gaussienne bidimentionnelle d'écart-type sigma
 */
void getDoGX(Mat& K, int w, double sigma)
{
  K= Mat(2*w+1,2*w+1, CV_64FC1);

  double alpha= 1/(2*M_PI*pow(sigma,4));
  double beta= -1/(2*sigma*sigma);
  
  for (int i=-w; i <= w; i++)
    for (int j=-w; j<= w; j++)
      K.at<double>(j+w,i+w)= i*alpha*exp(beta*(i*i+j*j));
}

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

int main( int argc, char** argv )
{
  Mat tmp = imread(argv[1]);
  Mat kx;

  if (tmp.channels() > 1)
    {
      cvtColor( tmp, src, COLOR_BGR2GRAY );
      tmp= src;
    }
  tmp.convertTo(src, CV_8UC1 );
  // attention, autre conversion en CV_32FC1 probablement nécessaire...

    //kx = (Mat_<double>(3,3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    //filter2D(src, src, -1, kx);
    Mat DyBis;
    Mat ky;
    Mat kybis;
    // Dx
    getDoGX(kx, 2, 1);
    filter2D(tmp, Dx, CV_64FC1, kx);
    //normalize(Dx, Dx, 255, 0, NORM_MINMAX, CV_8UC1);
    namedWindow("Dx");
    imshow("Dx", Dx);

    // Dy
    rotate(kx, ky, ROTATE_90_COUNTERCLOCKWISE);
    filter2D(tmp, Dy, CV_64FC1, ky);
    //normalize(Dy, Dy, 255, 0, NORM_MINMAX, CV_8UC1);
    namedWindow("Dy");
    imshow("Dy", Dy);

    // RESULTAT
    Mat result;
    add(Dx, Dy, result);
    //int w = 1;
    //copyMakeBorder(result, result, 2*w+1, 2*w+1, 2*w+1, 2*w+1, BORDER_CONSTANT, 0);
    namedWindow("Result");
    imshow("Result", result);

    //harris(Dx, Dy, );

  namedWindow("src");
  imshow("src", src);


  // Routine de traitement du clic souris
  setMouseCallback("src", mouse_callback, NULL);

  // A vous de compléter le code...

  //getDoGX();
  
  waitKey();
  return 0;
}



#include <iostream>
#include "image.hpp"
#include "sphere.hpp"
#include <ctime>

using namespace std;

int main(int argc, char** argv)
{
	int sampleSize = 64;
	int phong_exp = 1.0f;
	int d_sphere = 511;
	ImageData<float> em = ImageData<float>::load("GraceCathedral/grace_latlong.pfm");

    srand(time(0));
	int *asample = ImageData<float>::sample(em, sampleSize);
	srand(time(0));
	int *phsample = ImageData<float>::phongSample(em, phong_exp, sampleSize);

	Sphere s(d_sphere);
	ImageData<float> sphere_pfm = ImageData<float>::relightSphere(em, asample, sampleSize, s);

	ImageData<float>::tonMapping(em,1,2.2);
	
	ImageData<unsigned char> em_ppm_cdf = ImageData<float>::convert<unsigned char>(em);
	ImageData<unsigned char>::printSample(em_ppm_cdf, asample, sampleSize);

	ImageData<unsigned char> em_ppm_phong = ImageData<float>::convert<unsigned char>(em);
	ImageData<unsigned char>::printSample(em_ppm_phong, phsample, sampleSize);

	ImageData<float> em_pfm_phong = ImageData<unsigned char>::convert<float>(em_ppm_phong);
	ImageData<float> em_pfm_cdf = ImageData<unsigned char>::convert<float>(em_ppm_cdf);

	em_ppm_cdf.save("em_cdf");
	em_ppm_phong.save("em_phong");

	em_pfm_phong.save("em_phong");
	em_pfm_cdf.save("em_cdf");

	ImageData<float>::tonMapping(sphere_pfm,1,2.2);
	sphere_pfm.save("sphere");
	ImageData<unsigned char> sphere_ppm = ImageData<float>::convert<unsigned char>(sphere_pfm);
	sphere_ppm.save("sphere");

	system("pause");
    return 0;
}

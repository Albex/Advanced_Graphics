#ifndef __IMAGE__HPP__
#define __IMAGE__HPP__

#define _USE_MATH_DEFINES
#include <math.h>

#include <stdio.h>
#include <iostream>
#include <string>
#include <type_traits>
#include <stdint.h>
#include "sphere.hpp"

template <class T>
class ImageData {

public:
    ~ImageData() {
    }

    static ImageData<T> load(std::string path) {
	FILE          *imgFile;
	char          buf[1024];
	char          type;
	int           i, j, k, l;
	double        distance, x, y;
	unsigned int width, height, numComponents;
	T* imageData;
	T* tmpData;
	ImageData<T> img;

	imgFile= fopen( path.c_str(), "rb" );

	if (std::is_same<T, unsigned char>::value) {
	    if( fscanf( imgFile, "P%c \n", &type )!= 1 || type< '1' || type> '8')
	    {
		fclose( imgFile );

	    }
	    // skip comments
	    while( fscanf( imgFile, "#%[^\n]\n", buf ) )
		;
	    // read width
	    fscanf( imgFile, "%d", &width );
	    /* skip comments */
	    while( fscanf( imgFile, "#%[^\n]\n", buf ) )
		;
	    /* read height */
	    fscanf( imgFile, "%d", &height );
	    /* skip comments */
	    while( fscanf( imgFile, "#%[^\n]\n", buf ) )
		;
	    /* skip max. component and exactly one whitespace */
	    fscanf( imgFile, "%*d%*c" );

	    switch( type )
	    {
	    case '1': // ASCII bitmap
	    case '4': // binary bitmap
		std::cerr << "Bitmaps not implemented\n";
		fclose( imgFile );

	    case '2': // ASCII greymap
		imageData= new T[width*height*3];
		for( i= 0 ; i< height ; i++ )
		    for( j= 0 ; j< width ; j++ )
		    {
			fscanf( imgFile, "%d", &l );
			imageData[i*width+j]= l;
		    }
		numComponents= 1;
		break;
	    case '3': // ASCII RGB
		imageData= new T[width*height*3];
		for( i= 0 ; i< height ; i++ )
		    for( j= 0 ; j< width ; j++ )
			for( k= 0 ; k< 3 ; k++ )
			{
			    fscanf( imgFile, "%d", &l );
			    imageData[(i*width+j)*3+k]= l;
			}
		numComponents= 3;
		break;
	    case '5': // binary greymap
		imageData= new T[width*height];
		fread( imageData, 1, width*height, imgFile );
		numComponents= 1;
		break;
	    case '6': // binary RGB
		imageData= new T[width*height*3];
		fread( imageData, 1, width*height*3, imgFile );
		numComponents= 3;
		break;
	    }
	}
	else {
	    fscanf( imgFile, "PF \n", &type );
	    // skip comments
	    while( fscanf( imgFile, "#%[^\n]\n", buf ) )
		;

	    // read width
	    fscanf( imgFile, "%d", &width );
	    /* skip comments */
	    while( fscanf( imgFile, "#%[^\n]\n", buf ) )
		;

	    /* read height */
	    fscanf( imgFile, "%d", &height );
	    /* skip comments */
	    while( fscanf( imgFile, "#%[^\n]\n", buf ) )
		;



	    /* skip max. component and exactly one whitespace */
	    fscanf( imgFile, "%*f%*c" );



	    imageData= new T[width*height*3];

	    fread( imageData, sizeof(T), width*height*3, imgFile );
	    numComponents= 3;

	    fclose( imgFile );

	    tmpData= new T[width*3];

	    //invert image for reading!!!
	    for(int i=0;i<height/2;i++)
	    {
		for(int j=0;j<width;j++)
		{
		    for(int k=0;k<3;k++)
		    {
			int indexS = i*width*3 + j*3 + k;
			int indexD = (height-1 - i)*width*3 + j*3 + k;

			tmpData[j*3 + k] = imageData[indexS];
			imageData[indexS] = imageData[indexD];
			imageData[indexD] = tmpData[j*3 + k];
		    }
		}
	    }

	}
	delete[] tmpData;
	img.data = imageData;
	img.width = width;
	img.height = height;
	img.numComponents = numComponents;
	return img;
    }

    void save(std::string name) {
	FILE          *fp;
	char          buf[1024];
	char          type;
	int           i, j, k, l;
	double        distance, x, y;
	float max=1.0f;
	char space=' ';

// Write PGM image file with filename "file"

// The PGM file format for a GREYLEVEL image is:
// P5 (2 ASCII characters) <CR>
// two ASCII numbers for nx ny (number of rows and columns <CR>
// 255 (ASCII number) <CR>
// binary pixel data with one byte per pixel

// The PGM file format for a COLOR image is:
// P6 (2 ASCII characters) <CR>
// two ASCII numbers for nx ny (number of rows and columns <CR>
// 255 (ASCII number) <CR>
// binary pixel data with three bytes per pixel (one byte for each R,G,B)

	if (std::is_same<T, float>::value) {
	    fp=fopen((name + ".pfm").c_str(),"wb");
	    fputc('P', fp);
	    fputc('F', fp);
	    fputc(0x0a, fp);

	    fprintf(fp, "%d %d", width, height);
	    fputc(0x0a, fp);

	    fprintf(fp, "%f", -1.0f);
	    fputc(0x0a, fp);
	    for(i=height-1;i>=0;i--)
		fwrite(&data[i*width*numComponents],sizeof(float),width*numComponents, fp);
	}
	else {
	    fp=fopen((name + ".ppm").c_str(),"wb");
	    // write the first ASCII line with the file type
	    if(numComponents==1)
		fprintf(fp,"P5\n"); //greylevel image
	    else if(numComponents==3)
		fprintf(fp,"P6\n");  // color image

	    // write image dimensions
	    fprintf(fp,"%d %d\n",width,height);

	    // write '255' to the next line
	    fprintf(fp,"255\n");

	    fwrite(data,sizeof(unsigned char),width*height*numComponents, fp);

	}

	fclose(fp);
    }

    template <class T1>
    static ImageData<T1> convert(ImageData<T> img) {
	ImageData<T1> ret;
	T1 *img_out = new T1[img.width*img.height*img.numComponents];
	for (unsigned int i = 0 ; i < img.height ; ++i ) // height
	{
	    for ( int j = 0 ; j < img.width ; ++j ) // width
	    {
		//for ( int k = 0 ; k < img.numComponents ; ++k ) // color channels - 3 for RGB images
		//{
		    int index = i*img.width*img.numComponents + j*img.numComponents; //index within the image
		    if (std::is_same<T1, unsigned char>::value) {
			//typecast 0.0f -> 1.0f values to the 0 - 255 range
			img_out[index] = static_cast<unsigned char>(img.data[index]*255.0f); //R
			img_out[index + 1] = static_cast<unsigned char>(img.data[index + 1]*255.0f);//G
			img_out[index + 2] = static_cast<unsigned char>(img.data[index + 2]*255.0f);//B

		    }
		    else {
			//typecast 0 - 255 values to the 0.0f -> 1.0f range
			img_out[index] = static_cast<float>(img.data[index])/255.0f; //R
			img_out[index + 1] = static_cast<float>(img.data[index + 1])/255.0f;//G
			img_out[index + 2] = static_cast<float>(img.data[index + 2])/255.0f;//B
			}
		//}

	    }
	}
	ret.width = img.width;
	ret.height = img.height;
	ret.numComponents = img.numComponents;
	ret.data = img_out;
	return ret;
    }

	static void tonMapping(ImageData<float> img, int stops, float gamma) {
		for ( int i = 0 ; i < img.height ; ++i ) // height
		{
			for ( int j = 0 ; j < img.width ; ++j ) // width
			{
				int index = i*img.width*img.numComponents + j*img.numComponents; //index within the image
				float n_R = (img.data[index] * pow(2.0,stops) > 1.0) ? 1.0 : img.data[index] * pow(2,stops) ;
				float n_G = (img.data[index + 1] * pow(2.0,stops) > 1.0) ? 1.0 : img.data[index + 1] * pow(2,stops) ;
				float n_B = (img.data[index + 2] * pow(2.0,stops) > 1.0) ? 1.0 : img.data[index + 2] * pow(2,stops) ;
				img.data[index] = pow(n_R,1/gamma);
				img.data[index + 1] = pow(n_G, 1/gamma);
				img.data[index + 2] = pow(n_B, 1/gamma);
			}
		 }
	}

	static double xor128(void)
	{
		static uint32_t x = 123456789;
		static uint32_t y = 362436069;
		static uint32_t z = 521288629;
		static uint32_t w = 88675123;
		uint32_t t;
 
		t = x ^ (x << 11);
		x = y; y = z; z = w;
		w = w ^ (w >> 19) ^ (t ^ (t >> 8));
		return w/(double) 0xFFFFFFFF;
	}

	static double pxor128(void)
	{
		static uint32_t x = 123456789;
		static uint32_t y = 362436069;
		static uint32_t z = 521288629;
		static uint32_t w = 88675123;
		uint32_t t;
 
		t = x ^ (x << 11);
		x = y; y = z; z = w;
		w = w ^ (w >> 19) ^ (t ^ (t >> 8));
		return w/(double) 0xFFFFFFFF;
	}

	static int* sample(ImageData<T> img, int sampleNum)
	{
		float *row_mean = new float[img.height];
		T *line = new T[img.width];
		int *sample = new int[2*sampleNum];

		for(int i=0; i<img.height; i++)
		{
			row_mean[i] = 0;
			for(int j=0; j<img.width; j++)
			{
				int index = i*img.width*img.numComponents + j*img.numComponents;
				row_mean[i] += (img.data[index]+img.data[index+1]+img.data[index+2])/3.0;
			}
		}

		row_mean[0] = row_mean[0]*sin(M_PI)/(double) img.width;
		for(int i=1; i<img.height; i++)
		{
			row_mean[i] = (row_mean[i]*sin(M_PI*i/(double) img.height)/(double) img.width) + row_mean[i-1];
		}

		int j;
		float current_vsample;
		for(int i=0; i<sampleNum; i++)
		{
			//current_vsample = rand() % (int) ceil(row_mean[img.height-1]*100);
			current_vsample = ImageData<float>::xor128()*row_mean[img.height-1];
			//current_vsample = current_vsample / 100.0;

			j=0;
			while(current_vsample>row_mean[j] && j<img.height)
			{
				j++;
			}

			((current_vsample - row_mean[j-1]) < (row_mean[j] - current_vsample) && j != 0) ? j-- : j;
            sample[2*i] = j;

			int index = j*img.width*img.numComponents;
			line[0] = (img.data[index] + img.data[index + 1] + img.data[index + 2])/3.0;
			for(int l=1; l<img.width; l++)
			{
				index = j*img.width*img.numComponents + l*img.numComponents;
				line[l] = line[l-1] + (img.data[index] + img.data[index + 1] + img.data[index + 2])/3.0f;
			}

			//double current_hsample = rand() % (int) ceil(line[img.width-1]*100);
			//current_hsample = current_hsample / 100.0
			double current_hsample = ImageData<float>::xor128()*line[img.width-1];


			int k = 0;
			while(current_hsample > line[k] && k<img.width)
			{
				k++;
			}
			((current_hsample - line[k-1]) < (line[k] - current_hsample)) ? k-- : k;

			sample[2*i+1] = k;
		}

		return sample;
	}

	static int* phongSample(ImageData<T> img, double exponent, int numSample)
	{
		int *sample = new int[2*numSample];

		for(int i = 0; i<numSample; i++)
		{
			double sample1 = ImageData<float>::pxor128();
			double sample2 = ImageData<float>::pxor128();

			double theta = acos(pow(1-sample1, 1/(exponent+1)));
			double phi = 2*M_PI*sample2;

			sample[2*i+1] = floor(phi/(2 * M_PI) * img.width);
			sample[2*i] = floor(theta/M_PI * img.height);
		}
		
		return sample;
	}

	static void printSample(ImageData<T> img, int* sample, int sampleSize)
	{
		ImageData<T> printSample;

		for(int i=0; i<sampleSize; i++)
		{
			int sampleOrd = sample[2*i];
			int sampleAbs = sample[2*i+1];

			for(int j=0; j<5; j++)
			{
				int l = j-2;
				for(int k=0; k<5; k++)
				{
					int m = k-2;
					if (sampleOrd+l >= 0 && sampleOrd+l < img.height && sampleAbs+m >= 0 && sampleAbs+m < img.width)
					{
						int index = (sampleOrd+l)*img.width*img.numComponents + (sampleAbs+k)*img.numComponents;
						img.data[index] = 0;
						img.data[index + 1] = 255;
						img.data[index + 2] = 0;
					}
				}
			}
		}
	}

	static float integralLight(ImageData<float> img)
	{
		float L = 0;

		for(int i=0; i<img.height; i++)
		{
			for(int j=0; j<img.width; j++)
			{
				int indice = i*img.width*img.numComponents + j*img.numComponents;
				float theta = i * M_PI /(float) (img.height - 1);

				L += ((img.data[indice] + img.data[indice + 1] + img.data[indice + 2]) * sin(theta)/ 3.0f);
			}
		}

		L /= ((float) (img.height*img.width));

		return L;
	}

	static ImageData<float> relightSphere(ImageData<float> const &img, int* sample, int sampleSize, Sphere const &sphere_model)
	{
		ImageData<float> sph;
		float *sphere_out = new float[sphere_model.diameter*sphere_model.diameter*img.numComponents];
		float *r = new float[3];

		int indice_s, indice_i;
		float theta, phi, dot_p;
		float *s = new float[3];
		float *v = new float[3];
		v[0] = 0.0f;
		v[1] = -1.0f;
		v[2] = 0.0f;

		float L = integralLight(img);
		float cons = L/(M_PI * sampleSize);

		for(int i=0; i<sphere_model.diameter; i++)
		{
			for(int j=0; j<sphere_model.diameter; j++)
			{
				indice_s = i*sphere_model.diameter*img.numComponents + j*img.numComponents;

				sphere_out[indice_s] = 0.0f;
				sphere_out[indice_s + 1] = 0.0f;
				sphere_out[indice_s + 2] = 0.0f;

				// Reflectance
				dot_p = dot(v, sphere_model.normals + indice_s);
				r[0] = (2.0f * dot_p * sphere_model.normals[indice_s] - v[0]);
				r[1] = (2.0f * dot_p * sphere_model.normals[indice_s + 1] - v[1]);
				r[2] = (2.0f * dot_p * sphere_model.normals[indice_s + 2] - v[2]);

				float norm = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);

				r[0] /= norm;
				r[1] /= norm;
				r[2] /= norm;

				if (pow(i-sphere_model.diameter/2.0f, 2.0f) + pow(j-sphere_model.diameter/2.0f, 2.0f) <= pow(sphere_model.diameter/2.0f, 2.0f))
				{
					for(int k=0; k<sampleSize; k++)
					{
						indice_i = sample[2*k]*img.width*img.numComponents + sample[2*k + 1]*img.numComponents;

						theta = static_cast<float> (sample[2*k] * M_PI / (img.height));
						phi = static_cast<float> (sample[2*k+1] * 2*M_PI / (img.width));
						s[0] = sin(theta)*cos(phi);
						s[1] = sin(theta)*sin(phi);
						s[2] = cos(theta);

						float normalize = sqrt(img.data[indice_i]*img.data[indice_i]  + img.data[indice_i + 1]*img.data[indice_i + 1] + img.data[indice_i + 2]*img.data[indice_i + 2]);

						float cos_theta = dot(sphere_model.normals + indice_s, s);
						cos_theta = (cos_theta < 0.0f) ? 0.0f: cos_theta;

						sphere_out[indice_s] += (cos_theta * img.data[indice_i] / normalize);
						sphere_out[indice_s+1] += (cos_theta * img.data[indice_i+1] / normalize);
						sphere_out[indice_s+2] += (cos_theta * img.data[indice_i+2] / normalize);
					}
					sphere_out[indice_s] *= cons;
					sphere_out[indice_s + 1] *= cons;
					sphere_out[indice_s + 2] *= cons;
				}
			}
		}

		sph.height = sphere_model.diameter;
		sph.width = sphere_model.diameter;
		sph.numComponents = img.numComponents;
		sph.data = sphere_out;
		
		return sph;
	}

	static float ImageData<T>::dot(float* vec1, float* vec2) {
		return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2];
	}

    ImageData() {
	}

    ImageData(int width, int height, int numComponents, T* data) {
	this->width = width;
	this->height = height;
	this->numComponents = numComponents;
	this->data = data;

    }

//protected:
    T* data;
    unsigned int width;
    unsigned int height;
    unsigned int numComponents;
};

#endif

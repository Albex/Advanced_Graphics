#ifndef __SPHERE__HPP__
#define __SPHERE__HPP__

class Sphere {
public:
    Sphere(int rayon);
    ~Sphere();

    float* normals;
    int diameter;

private:
    float norm(float value);
};

#endif

// Tampere University
// TIE-52306 Computer Graphics Coding Assignment 2019
//
// Write your name and student id here:
//   Khoa Nguyen, khoa.nguyen@tuni.fi
//
// Mark here with an X what functionalities you implemented
// Note that different functionalities are worth different amount of points.
//
// Name of the functionality      |Done| Notes
//-------------------------------------------------------------------------------
// example functionality          |  X  | Example note: control this with var YYYY
// Mandatory functionalities ----------------------------------------------------
//   Perspective projection       |  X  | 
//   Phong shading                |  X  | 
//   Camera movement and rotation |  X  | 
//   Sharp shadows                |  X  | 
// Extra functionalities --------------------------------------------------------
//   Attend visiting lecture 1    |  X  | 
//   Attend visiting lecture 2    |  X  | 
//   Tone mapping                 |  X  | 
//   PBR shading                  |     | 
//   Soft shadows                 |  X  | 
//   Sharp reflections            |  X  | 
//   Glossy reflections           |     | 
//   Refractions                  |     | 
//   Caustics                     |     | 
//   SDF Ambient Occlusions       |  X  | 
//   Texturing                    |     | 
//   Simple game                  |     | 
//   Progressive path tracing     |     | 
//   Basic post-processing        |     | 
//   Advanced post-processing     |     | 
//   Screen space reflections     |     | 
//   Screen space AO              |  X  | 
//   Simple own SDF               |  X  | 
//   Advanced own SDF             |     | 
//   Animated SDF                 |  X  | 
//   Other?                       |     | 


#ifdef GL_ES
precision mediump float;
#endif

#define PI 3.14159265359
#define EPSILON 0.00001

// These definitions are tweakable.

/* Minimum distance a ray must travel. Raising this value yields some performance
 * benefits for secondary rays at the cost of weird artefacts around object
 * edges.
 */
#define MIN_DIST 0.08
/* Maximum distance a ray can travel. Changing it has little to no performance
 * benefit for indoor scenes, but useful when there is nothing for the ray
 * to intersect with (such as the sky in outdoors scenes).
 */
#define MAX_DIST 20.0
/* Maximum number of steps the ray can march. High values make the image more
 * correct around object edges at the cost of performance, lower values cause
 * weird black hole-ish bending artefacts but is faster.
 */
#define MARCH_MAX_STEPS 128
/* Typically, this doesn't have to be changed. Lower values cause worse
 * performance, but make the tracing stabler around slightly incorrect distance
 * functions.
 * The current value merely helps with rounding errors.
 */
#define STEP_RATIO 0.999
/* Determines what distance is considered close enough to count as an
 * intersection. Lower values are more correct but require more steps to reach
 * the surface
 */
#define HIT_RATIO 0.001

// Resolution of the screen
uniform vec2 u_resolution;

// Mouse coordinates
uniform vec2 u_mouse;

// Time since startup, in seconds
uniform float u_time;

struct material
{
    // The color of the surface
    vec4 color;
    // You can add your own material features here!
};

// Good resource for finding more building blocks for distance functions:
// http://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm

/* Basic box distance field.
 *
 * Parameters:
 *  p   Point for which to evaluate the distance field
 *  b   "Radius" of the box
 *
 * Returns:
 *  Distance to the box from point p.
 */
float box(vec3 p, vec3 b)
{
    vec3 d = abs(p) - b;
    return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}


/* Basic sd torus distance field.
 *
 * Parameters:
 *  p   Point for which to evaluate the distance field
 *  t   
 *
 * Returns:
 *  Distance to the torus from point p.
 */
float sdTorus( vec3 p, vec2 t )
{
  vec2 q = vec2(length(p.xz)-t.x,p.y);
  return length(q)-t.y;
}


float opSmoothUnion( float d1, float d2, float k )
{
    float h = max(k-abs(d1-d2),0.0);
    return min(d1, d2) - h*h*0.25/k;
	//float h = clamp( 0.5 + 0.5*(d2-d1)/k, 0.0, 1.0 );
	//return mix( d2, d1, h ) - k*h*(1.0-h);
}


/* Rotates point around origin along the X axis.
 *
 * Parameters:
 *  p   The point to rotate
 *  a   The angle in radians
 *
 * Returns:
 *  The rotated point.
 */
vec3 rot_x(vec3 p, float a)
{
    float s = sin(a);
    float c = cos(a);
    return vec3(
        p.x,
        c*p.y-s*p.z,
        s*p.y+c*p.z
    );
}

/* Rotates point around origin along the Y axis.
 *
 * Parameters:
 *  p   The point to rotate
 *  a   The angle in radians
 *
 * Returns:
 *  The rotated point.
 */
vec3 rot_y(vec3 p, float a)
{
    float s = sin(a);
    float c = cos(a);
    return vec3(
        c*p.x+s*p.z,
        p.y,
        -s*p.x+c*p.z
    );
}

/* Rotates point around origin along the Z axis.
 *
 * Parameters:
 *  p   The point to rotate
 *  a   The angle in radians
 *
 * Returns:
 *  The rotated point.
 */
vec3 rot_z(vec3 p, float a)
{
    float s = sin(a);
    float c = cos(a);
    return vec3(
        c*p.x-s*p.y,
        s*p.x+c*p.y,
        p.z
    );
}


/* Translate point around origin along the Z axis.
 *
 * Parameters:
 *  p   The point to rotate 
 *  d   The distance
 *
 * Returns:
 *  The rotated point.
 */
vec3 translate_z(vec3 p, float d)
{
    return vec3(
        p.x,
        p.y,
        p.z + d
    );
}

/* Each object has a distance function and a material function. The distance
 * function evaluates the distance field of the object at a given point, and
 * the material function determines the surface material at a point.
 */

float blob_distance(vec3 p)
{
    vec3 q = p - vec3(-0.5, -2.2 + abs(sin(u_time*3.0)), 2.0);
    return length(q) - 0.5 + sin(10.0*q.x)*sin(10.0*q.y)*sin(10.0*q.z)*0.07;
}

material blob_material(vec3 p)
{
    material mat;
    mat.color = vec4(1.0, 0.0275, 0.9529, 0.0);
    return mat;
}

float sphere_distance(vec3 p)
{
    return length(p - vec3(1.5, -1.8, 4.0)) - 1.2;
}

material sphere_material(vec3 p)
{
    material mat;
    mat.color = vec4(0.3725, 0.8549, 0.051, 1.0);
    return mat;
}

float room_distance(vec3 p)
{
    return max(
        -box(p-vec3(0.0,3.0,3.0), vec3(0.5, 0.5, 0.5)),
        -box(p-vec3(0.0,0.0,0.0), vec3(3.0, 3.0, 6.0))
    );
}

material room_material(vec3 p)
{
    material mat;

    mat.color = vec4(0.0, 0.0, 1.0, 1.0);
    
    if(p.y <= -2.98) mat.color.rgb = vec3(0.9216, 0.9216, 0.9216); // floor
    else if (p.y >= 2.98) mat.color.rgb = vec3(0.4784, 0.8863, 0.0157);  // roof

    if(p.x <= -2.98) mat.color.rgb = vec3(1.0, 0.0, 0.0); // left wall
    else if(p.x >= 2.98) mat.color.rgb = vec3(0.0, 1.0, 0.0); // right wall

    return mat;
}

float crate_distance(vec3 p)
{
    return box(rot_y(p-vec3(-1,-1,5), u_time), vec3(1, 2, 1));
}

material crate_material(vec3 p)
{
    material mat;
    mat.color = vec4(1.0, 1.0, 1.0, 1.0);

    vec3 q = rot_y(p-vec3(-1,-1,5), u_time) * 0.98;
    if(fract(q.x + floor(q.y*2.0) * 0.5 + floor(q.z*2.0) * 0.5) < 0.5)
    {
        mat.color.rgb = vec3(0.0, 0.0, 0.0);
    }
    return mat;
}


float torus_distance(vec3 p){
    return sdTorus(rot_x(p-vec3(-0.33, -0.4, -0.6), 8.*u_time), 
                    vec2(0.5, 0.05));
}

    material torus_material(vec3 p) {
    material mat;
    mat.color = vec4(0.749, 0.8667, 0.0745, .0);
    return mat;
}

/* The distance function collecting all others.
 *
 * Parameters:
 *  p   The point for which to find the nearest surface
 *  mat The material of the nearest surface
 *
 * Returns:
 *  The distance to the nearest surface.
 */
float map(
    in vec3 p,
    out material mat
){
    float min_dist = MAX_DIST*2.0;
    float dist = 0.0;

    dist = blob_distance(p);
    if(dist < min_dist) {
        mat = blob_material(p);
        min_dist = dist;
    }

    dist = room_distance(p);
    if(dist < min_dist) {
        mat = room_material(p);
        min_dist = dist;
    }

    dist = crate_distance(p);
    if(dist < min_dist) {
        mat = crate_material(p);
        min_dist = dist;
    }

    dist = sphere_distance(p);
    if(dist < min_dist) {
        mat = sphere_material(p);
        min_dist = dist;
    }

    // Add your own objects here!
    dist = torus_distance(p);
    if(dist < min_dist) {
        mat = torus_material(p);
        min_dist = dist;
    }

    return min_dist;
}

/* Calculates the normal of the surface closest to point p.
 *
 * Parameters:
 *  p   The point where the normal should be calculated
 *  mat The material information, produced as a byproduct
 *
 * Returns:
 *  The normal of the surface.
 *
 * See http://www.iquilezles.org/www/articles/normalsSDF/normalsSDF.htm if
 * you're interested in how this works.
 */
vec3 normal(vec3 p, out material mat)
{
    const vec2 k = vec2(1.0, -1.0);
    return normalize(
        k.xyy * map(p + k.xyy * EPSILON, mat) +
        k.yyx * map(p + k.yyx * EPSILON, mat) +
        k.yxy * map(p + k.yxy * EPSILON, mat) +
        k.xxx * map(p + k.xxx * EPSILON, mat)
    );
}

// ------------------- Phong Shading -------------------------
const vec3 lightColor = vec3(0.6392, 0.2224, 0.3824);
const vec3 AMBIENT = vec3(0.5255, 0.2137, 0.569);
const vec3 DIFFUSE = vec3(1.0, 1.0, 1.0);
const vec3 SPECULAR = vec3(0.0, 0.0, 0.0);
const float DIFFUSE_INTENSITY = 2.5;
const float SPECULAR_INTENSITY = 1.0;
const float SHININESS = 1.0;

// n = surface normal
// rd = view direction
// ld = light direction
vec3 phong_shading(vec3 n, vec3 rd, vec3 ld){
    // find the diffuse term
    float tempDiff = max(dot(-ld, n), 0.0);
    vec3 diffuse = tempDiff * DIFFUSE_INTENSITY * DIFFUSE;
    
    //find the specular term
    vec3 reflectionDir = reflect(-ld, n);
    float tempSpec = pow(dot(reflectionDir, rd), SHININESS);
    vec3 specular = tempSpec * SPECULAR * SHININESS;
    
    // add them together to create the Phong Shading
    vec3 color = AMBIENT + diffuse + specular;
    return color;
}


// ------------------------- Tone Mapping ------------------------------
float gamma = 2.2;
vec3 Uncharted2ToneMapping(vec3 color)
{
	float A = 0.15;
	float B = 0.50;
	float C = 0.10;
	float D = 0.20;
	float E = 0.02;
	float F = 0.30;
	float W = 11.2;
	float exposure = 1.6;
	color *= exposure;
	color = ((color * (A * color + C * B) + D * E) / 
                (color * (A * color + B) + D * F)) - E / F;
	float white = ((W * (A * W + C * B) + D * E) / 
                (W * (A * W + B) + D * F)) - E / F;
	color /= white;
	color = pow(color, vec3(1. / gamma));
	return color;
}


// -------------------- Soft shadow ---------------------
float calcSoftshadow( 
    in vec3 ro, 
    in vec3 rd, 
    in float mint, 
    in float tmax, 
    material mat )
{
    // bounding volume
    float tp = (0.8-ro.y)/rd.y; if( tp>0.0 ) tmax = min( tmax, tp );

    float res = 1.0;
    float t = mint;
    for( int i=0; i<16; i++ )
    {
		float h = map( ro + rd*t, mat);
        float s = clamp(8.0*h/t,0.0,1.0);
        res = min( res, s*s*(3.0-2.0*s) );
        t += clamp( h, 0.02, 0.10 );
        if( res<0.005 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}


// ------------------- Screen Space Ambient Occlusion ---------------------
float calcAO( in vec3 pos, in vec3 nor, material mat )
{
	float occ = 0.0;
    float sca = 1.0;
    for( int i=0; i<5; i++ )
    {
        float hr = 0.01 + 0.*float(i)/4.0;
        vec3 aopos =  nor * hr + pos;
        float dd = map( aopos , mat);
        occ += -(dd-hr)*sca;
        sca *= 0.95;
    }
    return clamp( 1.0 - 3.0*occ, 0.0, 1.0 ) * (0.5+0.5*nor.y);
}


// ------------------ Sphere Occlusion -----------------------

// Sphere occlusion
float sphOcclusion( in vec3 pos, in vec3 nor, in vec4 sph )
{
    vec3  di = sph.xyz - pos;
    float l  = length(di);
    float nl = dot(nor,di/l);
    float h  = l/sph.w;
    float h2 = h*h;
    float k2 = 1.0 - h2*nl*nl;

    // above/below horizon: Quilez - http://iquilezles.org/www/articles/sphereao/sphereao.htm
    float res = max(0.0,nl)/h2;
    // intersecting horizon: Lagarde/de Rousiers - http://www.frostbite.com/wp-content/uploads/2014/11/course_notes_moving_frostbite_to_pbr.pdf
    if( k2 > 0.0 ) 
    {
        #if 1
            res = nl*acos(-nl*sqrt( (h2-1.0)/(1.0-nl*nl) )) - sqrt(k2*(h2-1.0));
            res = res/h2 + atan( sqrt(k2/(h2-1.0)));
            res /= 3.141593;
        #else
            // cheap approximation: Quilez
            res = pow( clamp(0.5*(nl*h+1.0)/h2,0.0,1.0), 1.5 );
        #endif
    }

    return res;
}


/* Finds the closest intersection of the ray with the scene.
 *
 * Parameters:
 *  o           Origin of the ray
 *  v           Direction of the ray
 *  max_dist    Maximum distance the ray can travel. Usually MAX_DIST.
 *  p           Location of the intersection
 *  n           Normal of the surface at the intersection point
 *  mat         Material of the intersected surface
 *  inside      Whether we are marching inside an object or not. Useful for
 *              refractions.
 *
 * Returns:
 *  true if a surface was hit, false otherwise.
 */
bool intersect(
    in vec3 o,
    in vec3 v,
    in float max_dist,
    out vec3 p,
    out vec3 n,
    out material mat,
    bool inside,
    out float tt
) {
    float t = MIN_DIST;
    float dir = inside ? -1.0 : 1.0;
    bool hit = false;

    for(int i = 0; i < MARCH_MAX_STEPS; ++i)
    {
        p = o + t * v;
        float dist = dir * map(p, mat);
        
        hit = abs(dist) < HIT_RATIO * t;

        if(hit || t > max_dist) break;

        t += dist * STEP_RATIO;
    }
    tt = t;
    n = normal(p, mat);

    return hit;
}

/* Calculates the color of the pixel, based on view ray origin and direction.
 *
 * Parameters:
 *  o   Origin of the view ray
 *  v   Direction of the view ray
 *
 * Returns:
 *  Color of the pixel.
 */
vec3 render(vec3 o, vec3 v)
{
    // This lamp is positioned at the hole in the roof.
    vec3 lamp_pos = vec3(0.0, 3.1, 3.0);
    vec3 ld = normalize(lamp_pos); // light direction

    vec3 p, n;
    material mat;
    float tt;
    // Compute intersection point along the view ray.
    bool hit = intersect(o, v, MAX_DIST, p, n, mat, false, tt);

    // 
    if (hit) {
        // mat.color.rgb *= phong_shading(normal(p, mat), v, ld);
        
        // sharp shadow 
        // vec3 pp = p;
        // vec3 nn;
        // material matmat;
        // hit = intersect(pp+0.001*nn, ld, MAX_DIST, pp, nn, matmat, hit, tt);
        // if (hit) {
        //     mat.color.rgb *= 0.4;
        // }

        // lighting
        float occ = calcAO(p, n, mat);
        vec3  hal = normalize(o - ld);
        float amb = sqrt(clamp( 0.5+0.5*n.y, 0.0, 1.0 ));
        float dif = clamp( dot( n, ld ), 0.0, 1.0 );
        float bac = clamp( dot( n, 
                        normalize(vec3(-ld.x,0.0,-ld.z))), 0.0, 1.0 )
                        *clamp( 1.0-p.y,0.0,1.0);
        vec3 ref = reflect(v, n);
        float dom = smoothstep( -0.2, 0.2, ref.y );
        float fre = pow( clamp(1.0+dot(n, v),0.0,1.0), 2.0 );
        dif *= calcSoftshadow( p, ld, 0.02, 2.5, mat);
        dom *= calcSoftshadow( p, ref, 0.02, 2.5, mat );
		float spe = pow( clamp( dot( n, hal ), 0.0, 1.0 ),16.0)*
                    dif *
                    (0.04 + 0.96*pow( clamp(1.0+dot(hal,v),0.0,1.0), 5.0 ));
        vec3 lin = vec3(0.0);
        lin += 3.80*dif*vec3(1.30,1.00,0.70);
        lin += 1.66*amb*vec3(0.40,0.60,1.15)*occ;
        lin += 0.85*dom*vec3(0.40,0.60,1.30)*occ;
        lin += 1.55*bac*vec3(0.25,0.25,0.25)*occ;
        lin += 0.25*fre*vec3(1.00,1.00,1.00)*occ;

        mat.color.rgb *= lin;
        mat.color.rgb += 7.00*spe*vec3(1.10,0.90,0.70);

        float sdfOcc = sphOcclusion(p, n, mat.color);

        mat.color.rgb= mix( mat.color.rgb, vec3(0.7,0.7,0.9),
                    1.0-exp( -0.0001*tt*tt*tt)) + 5.0*sdfOcc;
    }
    
    // tone mapping
    mat.color.rgb *= Uncharted2ToneMapping(mat.color.rgb);
    return mat.color.rgb;
}


void main()
{
    // This is the position of the pixel in normalized device coordinates.
    vec2 uv = (gl_FragCoord.xy/u_resolution)*2.0-1.0;
    // Calculate aspect ratio
    float aspect = u_resolution.x/u_resolution.y;

    // Modify these two to create perspective projection!
    // Origin of the view ray
    vec3 o = vec3(0.0, 1.0, -6.0);

    // Direction of the view ray
    vec3 v = normalize(vec3(uv.x * aspect, uv.y, 1.5));
    // v = rot_z(v, 0.15*sin(u_time));
    // v = rot_y(v, -0.15*cos(u_time));
    // v = translate_z(v, 0.04*sin(u_time));
        
    gl_FragColor = vec4(render(o, v), 1.0);
}

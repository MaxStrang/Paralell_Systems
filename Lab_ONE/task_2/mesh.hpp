#pragma once

#include "common.hpp"

#include <cassert>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

// return square of x
static inline float sq(const float &x)
{
	return x * x;
}

// return square distance between two d-dimensional points
template <int d> static inline float sqdst(const point<d> &a, const point<d> &b)
{
	float sum = 0.0f;
	for (int i = 0; i < d; ++i)
		sum += sq(a.v[i] - b.v[i]);
	return sum;
}

template <const int d> struct mesh
{
	mesh(const int n) : n(n)
	{
		assert(n > 0 && n % 16 == 0);
		for(int i = 0; i < d; i++)
		{
			soa_data[i] = (float *)aligned_alloc(32, sizeof(float)*n);
			assert(soa_data[i]);
		}
	}
	~mesh()
	{
		for(int i = 0; i < d; i++)
		{
			free(soa_data[i]);
		}
	}
	void set(const point<d> p, int i)
	{
		assert(i >= 0 && i < n);
		for(int j = 0; j < d; j++)
		{
			soa_data[j][i] = p.v[j];
		}
	}
	ball<d> calc_ball()
	{
		ball<d> b;

		float min[d], max[d];
		for(int j = 0; j < d; j++)
		{
			__m256 vmin = _mm256_set1_ps(soa_data[j][0]);
			__m256 vmax = _mm256_set1_ps(soa_data[j][0]);
			for(int i = 0; i < n; i += 8)
			{
				__m256 values = _mm256_load_ps(&soa_data[j][i]);
				vmin = _mm256_min_ps(vmin, values);
				vmax = _mm256_max_ps(vmax, values);
			}
			min[j] = horizontal_min(vmin);
			max[j] = horizontal_max(vmax);
			b.center.v[j] = (max[j] - min[j]) * 0.5f + min[j];
		}
		__m256 maxsqdst = _mm256_setzero_ps();
		__m256 center[d];
		for(int j = 0; j < d; j++)
		{
			center[j] = _mm256_set1_ps(b.center.v[j]);
		}
		for(int i = 0; i < n; i += 8)
		{
			__m256 sqdist = _mm256_setzero_ps();
			for(int j = 0; j < d; j++)
			{
				__m256 diff = _mm256_sub_ps(_mm256_load_ps(&soa_data[j][i]), center[j]);
				sqdist = _mm256_add_ps(sqdist, _mm256_mul_ps(diff, diff));
			}
		maxsqdst = _mm256_max_ps(maxsqdst, sqdist);
		}
		b.radius = sqrtf(horizontal_max(maxsqdst));
		return b;
	}
	int farthest(const point<d>& p)
	{
		int argmax = 0;
		float maxsqdst = 0.0f;

		for(int i = 0; i < n; i++)
		{
			float sqdist = 0.0f;
			for(int j = 0; j < d; j++)
			{
				float diff = soa_data[j][i] - p.v[j];
				sqdist += diff * diff;
			}
			if(sqdist > maxsqdst)
			{
				maxsqdst = sqdist;
				argmax = i;
			}
		}
	return argmax;
	}
	private:
		const int n;
		float *soa_data[d];
		//Find horizontal minimum of __m256 reg
		float horizontal_min(__m256 vec)
		{
			alignas(32) float tmp[8];
			_mm256_store_ps(tmp, vec);
			float result = tmp[0];
			for(int i = 1; i < 8; i++)
			{
				result = std::max(result, tmp[i]);
			}
		return result;
		}
		float horizontal_max(__m256 vec)
		{
			alignas(32) float tmp[8];
			_mm256_store_ps(tmp, vec);
			float result = tmp[0];
			for(int i = 1; i < 8; i++)
			{
				result = std::max(result, tmp[i]);
			}
		return result;
		}
};

// mesh of n d-dimensional points stored according to the AoS layout
/*template <const int d> struct mesh
{
	// constructor
	mesh(const int n) : n(n)
	{
		assert(n > 0 && n % 16 == 0);
		data = (point<d> *)malloc(sizeof(point<d>) * n);
		assert(data);
	}
	// destructor
	~mesh()
	{
		free(data);
	}
	// set a point of the mesh
	void set(const point<d> p, int i)
	{
		assert(i >= 0 && i < n);
		data[i] = p;
	}
	// calculate center and radius of the ball enclosing the points
	ball<d> calc_ball()
	{
		ball<d> b;
		// calculate the center
		point<d> min = data[0];
		point<d> max = data[0];
		for (int i = 1; i < n; ++i)
			for (int j = 0; j < d; ++j)
				min.v[j] = data[i].v[j] < min.v[j] ? data[i].v[j] : min.v[j],
				max.v[j] = data[i].v[j] > max.v[j] ? data[i].v[j] : max.v[j];
		for (int i = 0; i < d; ++i)
			b.center.v[i] = (max.v[i] - min.v[i]) * 0.5f + min.v[i];
		// calculate the radius
		float tmp, maxsqdst = 0.0f;
		for (int i = 0; i < n; ++i)
		{
			tmp = sqdst(data[i], b.center);
			maxsqdst = maxsqdst > tmp ? maxsqdst : tmp;
		}
		b.radius = sqrtf(maxsqdst);
		// return the enclosing ball
		return b;
	}
	// return the index of the farthest point from the given point p (OPTIONAL)
	int farthest(point<d> p)
	{
		int argmax = 0;
		float maxsqdst = sqdst(data[0], p);
		for (int i = 1; i < n; ++i)
		{
			float sqdsti = sqdst(data[i], p);
			if (sqdsti > maxsqdst)
			{
				maxsqdst = sqdsti;
				argmax = i;
			}
		}
		return argmax;
	}

private:
	const int n = 0;
	point<d> *data = nullptr;
};*/

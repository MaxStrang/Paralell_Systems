#pragma once

#include <cassert>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

static inline double sq(double x)
{
	return x*x;
}

struct vector
{
	//Constructor for AVX version
	vector(const int n) : n(n)
	{
		assert(n > 0 && n % 8 == 0);
		data = (double*)aligned_alloc(32, sizeof(double) * n);
		assert(data);
	}

	//Destructor for AVX version
	~vector()
	{
		free(data);
	}

	void set(double x, int i)
	{
		assert(i >= 0 && i < n);
		data[i] = x;
	}

	//Vectorized cosine similarity
	static double cosine_similarity(const vector &a, const vector&b)
	{
		assert (a.n == b.n);
		const int n = a.n;

		__m256d ab_sum = _mm256_setzero_pd(); //Ackumulator for A x B
		__m256d aa_sum = _mm256_setzero_pd(); //Ackumulator for A x A
		__m256d bb_sum = _mm256_setzero_pd(); //Ackumulator for B x B

		for(int i = 0; i < n; i += 4) //Loop through 4 elements at once, then move on
		{
			__m256d a_vals = _mm256_load_pd(&a.data[i]);
			__m256d b_vals = _mm256_load_pd(&b.data[i]);

			ab_sum = _mm256_fmadd_pd(a_vals, b_vals, ab_sum);
			aa_sum = _mm256_fmadd_pd(a_vals, a_vals, aa_sum);
			bb_sum = _mm256_fmadd_pd(b_vals, b_vals, bb_sum);
		}

		//Sum all four elements in each register
		double ab = hsum256(ab_sum);
		double aa = hsum256(aa_sum);
		double bb = hsum256(bb_sum);

		return ab / (std::sqrt(aa) * std::sqrt(bb));
	}

	private:
		const int n = 0;
		double *data = nullptr;

		//Here is the help-function to sumarize the elements in a reister
		static double hsum256(__m256d vec)
		{
			__m128d low = _mm256_castpd256_pd128(vec);
			__m128d high = _mm256_extractf128_pd(vec, 1);
			__m128d sum_low = _mm_hadd_pd(low, low);
			__m128d sum_high = _mm_hadd_pd(high, high);
			__m128d summary = _mm_hadd_pd(sum_low, sum_high);

			double result[2];
			_mm_storeu_pd(result, summary);
			return result[0];
		}
};

/*struct vector
{
	// constructor
	vector(const int n) : n(n)
	{
		assert(n > 0 && n % 8 == 0);
		data = (double*)malloc(sizeof(double) * n);
		assert(data);
	}
	// destructor
	~vector()
	{
		free(data);
	}
	// set a point of the mesh
	void set(double x, int i)
	{
		assert(i >= 0 && i < n);
		data[i] = x;
	}
	static double cosine_similarity(const vector &a,const vector &b) {
		assert(a.n==b.n);
		const int n = a.n;
		double ab = 0.0f;
		double aa = 0.0f;
		double bb = 0.0f;
		for(int i=0; i<n; ++i)
		{
			aa += sq(a.data[i]);
			bb += sq(b.data[i]);
			ab += a.data[i]*b.data[i];
		}
		return ab/(sqrtf(aa)*sqrtf(bb));
   }

private:
	const int n = 0;
	double *data = nullptr;
};
*/

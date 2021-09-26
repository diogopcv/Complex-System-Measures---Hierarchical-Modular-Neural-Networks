#ifndef _PROTOS_
#define _PROTOS_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

typedef struct connect{
	int posSyn, preSyn;
	float * weight;	
	int length;
	struct connect * right;
	struct connect * down;
} connect;

void calcCluster (connect * head, int nneuron, char * baseName, int trial);

void dijkstra (connect * head, int nneuron, char * baseName, int trial);

connect * createCortex (int nneuron, float rescaleFac, float sizeNet, int * nLs, unsigned char * typeList);

connect * createRand (int nneuron, int numConn);

connect * createRegNet (int nneuron, int numConn);

connect * createHMNet (int nneuron, int m, float prob, float probEx);

void desalocar(connect * head, int nneuron);

#endif

#include "protos.h"
#include "tools.h"

void mainFuncao(int sim);

int main(int argc, char * argv[]) {	
	int i, seed;
	for(i = 0; i < 30; i++){
		seed = -1089432789 + i*100000;
		setSeed(seed);
		mainFuncao(i+1);
	}
	return 0;
}

void mainFuncao(int sim){	
	
	connect * head;
	char * baseName;
	int nneuron = 1024;
	 
	head = createHMNet (nneuron, 0, 0.01, 0.9);
	baseName = "mod0";
	dijkstra(head, nneuron, baseName, sim);
	calcCluster(head, nneuron, baseName, sim);

	desalocar(head,nneuron);
	
	head = createHMNet (nneuron, 1, 0.01, 0.9);
	baseName = "mod1";
	dijkstra(head, nneuron, baseName, sim);
	calcCluster(head, nneuron, baseName, sim);

	desalocar(head,nneuron);
	
	head = createHMNet (nneuron, 2, 0.01, 0.9);
	baseName = "mod2";
	dijkstra(head, nneuron, baseName, sim);
	calcCluster(head, nneuron, baseName, sim);

	desalocar(head,nneuron);
	
	head = createHMNet (nneuron, 3, 0.01, 0.9);
	baseName = "mod3";
	dijkstra(head, nneuron, baseName, sim);
	calcCluster(head, nneuron, baseName, sim);

	desalocar(head,nneuron);
	
	head = createHMNet (nneuron, 4, 0.01, 0.9);
	baseName = "mod4";
	dijkstra(head, nneuron, baseName, sim);
	calcCluster(head, nneuron, baseName, sim);

	desalocar(head,nneuron);			

	return;
}

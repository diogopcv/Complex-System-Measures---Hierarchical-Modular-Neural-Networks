#include "protos.h"
#include "tools.h"

void dijkstra (connect * head, int V, char * baseName, int trial)
{
	char vis[V];
	int i, Vi, j;
	float dis[V], avgDis;
	connect * mtxPre;
	
	char subFile[15], nameFile[30];
	sprintf(subFile, "_path_%d.dat", trial);
	strcpy(nameFile, baseName);
	strcat(nameFile, subFile);
		
	FILE * pfile = fopen(nameFile,"w");
	
	for (Vi = 0; Vi < V; Vi++){
		mtxPre = head->right;
		avgDis = 0;
		memset (vis, 0, V*sizeof (char));
		memset (dis, 0x7f, V*sizeof (int));
		dis[Vi] = 0;
		connect * mtxPre = head->right, * mtxAux;
		while (1)
		{
			int i, n = -1;
			
			for (i = 0; i < V; i++)
				if (! vis[i] && (n < 0 || dis[i] < dis[n]))
					n = i;
			
			if (n < 0)
				break;
			vis[n] = 1;
			
			mtxAux = mtxPre[n].down;
			while (mtxAux->posSyn >= 0){
				if (dis[mtxAux->posSyn] > dis[n] + mtxAux->weight[0])
					dis[mtxAux->posSyn]  = dis[n] + mtxAux->weight[0];	
				mtxAux = mtxAux->down;
			}
		}
		for (j = 0; j < V; j++)
			fprintf(pfile,"%f\t", dis[j]);
		fprintf(pfile,"\n");
	}
	fclose(pfile);
	return ;
}

void calcCluster (connect * head, int nneuron, char * baseName, int trial){
	int i, j, h, count;
	float wij, wih, wji, wjh, whi, whj, clusterT = 0.0;
	float dt[nneuron], drec[nneuron], cluster[nneuron];
	connect * mtxAux, * mtxPos = head->down, * mtxPre = head->right;
	
	omp_set_num_threads(8);	
	
	for (i = 0; i <nneuron; i++){
		cluster[i] = 0.0;
		drec[i] = 0.0;
		dt[i] = 0.0;
	}
	
	for (i = 0; i <nneuron; i++){
		mtxAux = mtxPos[i].right;
		count = 0;
		while(count < mtxPos[i].length){
			dt[i] += 1;
			mtxAux = mtxAux->right;
			count++;
		}
		mtxAux = mtxPre[i].down;
		count = 0;
		while(count < mtxPre[i].length){
			dt[i] += 1;
			mtxAux = mtxAux->down;
			count++;			
		}
	}
	
	for (i = 0; i <nneuron; i++){
		for (j = 0; j <nneuron; j++){
			count = 0;
			wij = 0.0;
			mtxAux = mtxPre[i].down;
			while(count < mtxPre[i].length){
				if(mtxAux->posSyn == j){
					wij = 1;
					break;	
				}
				mtxAux = mtxAux->down;
				count++;
			}
			count = 0;
			wji = 0.0;
			mtxAux = mtxPre[j].down;
			while(count < mtxPre[j].length){
				if(mtxAux->posSyn == i){
					wji = 1;
					break;	
				}
				mtxAux = mtxAux->down;
				count++;			
			}
			drec[i] += wij*wji; 
		}	
	}
	
	#pragma omp parallel for private(j,h,mtxAux,count,wij, wih, wji, wjh, whi, whj)
	for (i = 0; i <nneuron; i++){
		for (j = 0; j <nneuron; j++){
			for (h = 0; h <nneuron; h++){
				count = 0;
				mtxAux = mtxPre[i].down;
				wij = 0.0, wih = 0.0;
				while(count < mtxPre[i].length){
					if(h >= j){
						if(mtxAux->posSyn == j)
							wij = 1;
						if(mtxAux->posSyn == h){
							wih = 1;
							break;	
						}
					}
					else{
						if(mtxAux->posSyn == h)
							wih = 1;
						if(mtxAux->posSyn == j){
							wij = 1;
							break;	
						}									
					}
					mtxAux = mtxAux->down;
					count++;
				}
				
				count = 0;
				mtxAux = mtxPre[j].down;
				wji = 0.0, wjh = 0.0;
				while(count < mtxPre[j].length){
					if(h >= i){
						if(mtxAux->posSyn == i)
							wji = 1;
						if(mtxAux->posSyn == h){
							wjh = 1;
							break;	
						}
					}
					else{
						if(mtxAux->posSyn == h)
							wjh = 1;
						if(mtxAux->posSyn == i){
							wji = 1;
							break;	
						}									
					}
					mtxAux = mtxAux->down;
					count++;				
				}
				
				count = 0;
				mtxAux = mtxPre[h].down;
				whi = 0.0, whj = 0.0;
				while(count < mtxPre[h].length){
					if(i >= j){
						if(mtxAux->posSyn == j)
							whj = 1;
						if(mtxAux->posSyn == i){
							whi = 1;
							break;	
						}
					}
					else{
						if(mtxAux->posSyn == i)
							whi = 1;
						if(mtxAux->posSyn == j){
							whj = 1;
							break;	
						}									
					}
					mtxAux = mtxAux->down;
					count++;				
				}
				wij = cbrt(wij);
				wji = cbrt(wji);
				wih = cbrt(wih);
				whi = cbrt(whi);
				wjh = cbrt(wjh);
				whj = cbrt(whj);
				cluster[i] +=  (wij + wji)*(wih + whi)*(wjh + whj);
			}
		}
	}
	
	char subFile[15], nameFile[30];
	sprintf(subFile, "_cluster_%d.dat", trial);
	strcpy(nameFile, baseName);
	strcat(nameFile, subFile);		
	FILE * pfile = fopen(nameFile,"w");

	for (i = 0; i <nneuron; i++){
		cluster[i] = cluster[i]/( 2 * ( dt[i] * ( dt[i] - 1 ) - 2 * drec[i] ) );
		fprintf(pfile, "%f\n", cluster[i]);
	}

	fclose(pfile);

	return;
}

connect * createCortex (int nneuron, float rescaleFac, float sizeNet, int * nLs, unsigned char * typeList){
	int count, length, i, j, k, p,listNeuron[20000];
	int xPre, yPre, xPos, yPos, pre, typePre, typePos, beginI, endI, numConn;
	int beginK, endK, initPre, initPos, nlayerPre, nlayerPos, layerPre, layerPos;
    double scaleL1, scaleL23, scaleL4, scaleL5, scaleL6, scalePre, scalePos;
	double sort, re, dist;    
   	connect * mtConn, * MtConnAux, * head;
   	connect * MtConnLine, * MtConnColumn;    
    
    //definindo escala por camada
    scaleL1 = sizeNet/nLs[0];
    scaleL23 = sizeNet/nLs[1];
    scaleL4 = sizeNet/nLs[2];
    scaleL5 = sizeNet/nLs[3];
    scaleL6 = sizeNet/nLs[4];
    
    //carrega arquivos de conexao e arborizacao axonal
    FILE * connFile, * axonFile;
    double connDat[33][21], axonDat[17][6];
    connFile = fopen("dataConn.dat","r");
    axonFile = fopen("dataAxon.dat","r");
    for(i = 0; i < 33*21; i++){
    	fscanf(connFile,"%lf", &connDat[i/21][i%21]);
    }
    for(i = 0; i < 17*6; i++){
    	fscanf(axonFile,"%lf", &axonDat[i/6][i%6]);
    	axonDat[i/6][i%6] *= rescaleFac;    	
    }
    fclose(connFile);
    fclose(axonFile);

    //Incializando lista encadeada circular e Definindo por camada, tipo de cada neurônio   
    head = (connect *) mallocc(sizeof(connect));
	MtConnColumn = (connect *) mallocc(nneuron * sizeof(connect));
	MtConnLine = (connect *) mallocc(nneuron * sizeof(connect));
	
	head->posSyn = -1;
	head->preSyn = -1;
	head->weight = NULL;  
	head->down = &MtConnColumn[0];
	head->right = &MtConnLine[0];
	
	count = 0;
    for (i = 0; i < nLs[0]*nLs[0]; i++) { 
    	typeList[count] = 0;    	
		MtConnColumn[count].posSyn = count;
		MtConnColumn[count].preSyn = -1;
		MtConnColumn[count].weight = NULL;
		MtConnColumn[count].length = 0;
		MtConnColumn[count].right = &MtConnColumn[count];
		MtConnColumn[count].down = &MtConnColumn[count+1];	
		MtConnLine[count].posSyn = -1;
		MtConnLine[count].preSyn = count;
		MtConnLine[count].weight = NULL;
		MtConnLine[count].length = 0;
		MtConnLine[count].down = &MtConnLine[count];
		MtConnLine[count].right = &MtConnLine[count+1];  	
    	count++;
    }
    
    for (i = 0; i < nLs[1]*nLs[1]; i++) { 
    	sort = next();
    	if (sort < 0.78)
	    	typeList[count] = 1;
    	else if (sort >= 0.78 && sort < 0.87)
	    	typeList[count] = 2;    		
    	else
	    	typeList[count] = 3;    		
		MtConnColumn[count].posSyn = count;
		MtConnColumn[count].preSyn = -1;
		MtConnColumn[count].weight = NULL;
		MtConnColumn[count].length = 0;
		MtConnColumn[count].right = &MtConnColumn[count];
		MtConnColumn[count].down = &MtConnColumn[count+1];	
		MtConnLine[count].posSyn = -1;
		MtConnLine[count].preSyn = count;
		MtConnLine[count].weight = NULL;
		MtConnLine[count].length = 0;
		MtConnLine[count].down = &MtConnLine[count];
		MtConnLine[count].right = &MtConnLine[count+1];  	
    	count++;	
    }
    
    for (i = 0; i < nLs[2]*nLs[2]; i++) { 
    	sort = next();
    	if (sort < 0.27)
	    	typeList[count] = 4;
    	else if (sort >= 0.27 && sort < 0.54)
	    	typeList[count] = 5;    		
    	else if (sort >= 0.54 && sort < 0.81)
	    	typeList[count] = 6;    		
    	else if (sort >= 0.81 && sort < 0.85)
	    	typeList[count] = 7;    		
    	else
	    	typeList[count] = 8;    		
		MtConnColumn[count].posSyn = count;
		MtConnColumn[count].preSyn = -1;
		MtConnColumn[count].weight = NULL;
		MtConnColumn[count].length = 0;
		MtConnColumn[count].right = &MtConnColumn[count];
		MtConnColumn[count].down = &MtConnColumn[count+1];	
		MtConnLine[count].posSyn = -1;
		MtConnLine[count].preSyn = count;
		MtConnLine[count].weight = NULL;
		MtConnLine[count].length = 0;
		MtConnLine[count].down = &MtConnLine[count];
		MtConnLine[count].right = &MtConnLine[count+1];  	
    	count++;   	
    }
    
    for (i = 0; i < nLs[3]*nLs[3]; i++) {
    	sort = next();
    	if (sort < 0.64)
	    	typeList[count] = 9;
    	else if (sort >= 0.64 && sort < 0.81)
	    	typeList[count] = 10;    		
    	else if (sort >= 0.81 && sort < 0.89)
	    	typeList[count] = 11;    		
    	else
	    	typeList[count] = 12;    		
		MtConnColumn[count].posSyn = count;
		MtConnColumn[count].preSyn = -1;
		MtConnColumn[count].weight = NULL;
		MtConnColumn[count].length = 0;
		MtConnColumn[count].right = &MtConnColumn[count];
		MtConnColumn[count].down = &MtConnColumn[count+1];	
		MtConnLine[count].posSyn = -1;
		MtConnLine[count].preSyn = count;
		MtConnLine[count].weight = NULL;
		MtConnLine[count].length = 0;
		MtConnLine[count].down = &MtConnLine[count];
		MtConnLine[count].right = &MtConnLine[count+1];  	
    	count++;	
    }
    
    for (i = 0; i < nLs[4]*nLs[4] - 1; i++) {
    	sort = next();
    	if (sort < 0.62)
	    	typeList[count] = 13;
    	else if (sort >= 0.62 && sort < 0.82)
	    	typeList[count] = 14;	
    	else if (sort >= 0.82 && sort < 0.91)
	    	typeList[count] = 15;	
    	else
	    	typeList[count] = 16;
		MtConnColumn[count].posSyn = count;
		MtConnColumn[count].preSyn = -1;
		MtConnColumn[count].weight = NULL;
		MtConnColumn[count].length = 0;
		MtConnColumn[count].right = &MtConnColumn[count];
		MtConnColumn[count].down = &MtConnColumn[count+1];	
		MtConnLine[count].posSyn = -1;
		MtConnLine[count].preSyn = count;
		MtConnLine[count].weight = NULL;
		MtConnLine[count].length = 0;
		MtConnLine[count].down = &MtConnLine[count];
		MtConnLine[count].right = &MtConnLine[count+1];  	
    	count++; 	
    }	

   	sort = next();
   	if (sort < 0.62)
    	typeList[count] = 13;
   	else if (sort >= 0.62 && sort < 0.82)
    	typeList[count] = 14;	
   	else if (sort >= 0.82 && sort < 0.91)
    	typeList[count] = 15;	
   	else
   		typeList[count] = 16;	
	MtConnColumn[count].posSyn = count;
	MtConnColumn[count].preSyn = -1;
	MtConnColumn[count].weight = NULL;
	MtConnColumn[count].length = 0;
	MtConnColumn[count].right = &MtConnColumn[count];
	MtConnColumn[count].down = &MtConnColumn[0];	
	MtConnLine[count].posSyn = -1;
	MtConnLine[count].preSyn = count;
	MtConnLine[count].weight = NULL;
	MtConnLine[count].length = 0;
	MtConnLine[count].down = &MtConnLine[count];
	MtConnLine[count].right = &MtConnLine[0];   	
	
	/* A info do arquivo de conexao é lida linha a linha (p). Para cada linha
	varre-se todos o neuronios, aqueles que pertencerem ao tipo especificado
	pela linha é analisadq a info da linha em questao. Com a leitura é coletado 
	o indice dos neuronios potencialmente pre-sinapticos (analisando o tipo e
	distância (arquivo de arborizacao axonal) ). Após isso é sorteado dessa lista
	a quantidade de conexoes especificadas no arquivo de conexao */   
	
	// varredura das linhas do arquivo de conexao
	for (p = 0; p < 33; p++){
		typePos = (int) connDat[p][0];
		layerPos = (int) connDat[p][1];	
		// varredura de todos o neuronios
		if (typePos == 0){
			beginI = 0;
			endI = nLs[0]*nLs[0];
		}
		else if (typePos == 1 || typePos == 2 || typePos == 3){
			beginI = nLs[0]*nLs[0];
			endI =  nLs[0]*nLs[0] + nLs[1]*nLs[1];				
		}
		else if (typePos == 4 || typePos == 5 || typePos == 6 || typePos == 7 || typePos == 8){
			beginI = nLs[0]*nLs[0] + nLs[1]*nLs[1];
			endI =  nLs[0]*nLs[0] + nLs[1]*nLs[1] + nLs[2]*nLs[2];		
		}
		else if (typePos == 9 || typePos == 10 || typePos == 11 || typePos == 12){
			beginI = nLs[0]*nLs[0] + nLs[1]*nLs[1] + nLs[2]*nLs[2];	
			endI = nLs[0]*nLs[0] + nLs[1]*nLs[1] + nLs[2]*nLs[2] + nLs[3]*nLs[3];		
		}
		else{
			beginI = nLs[0]*nLs[0] + nLs[1]*nLs[1] + nLs[2]*nLs[2] + nLs[3]*nLs[3];	
			endI = nneuron;	
		}		
		for (i = beginI; i < endI; i++){
			typePos = typeList[i];
			//verifica se neuronio pos-sinaptico é o tipo especificado na linha
			if (typePos == (int) connDat[p][0]){
				// define a camada, escala e offset na lista de neuronios respectivo ao neuronio pos em questao			
				if (typePos == 0){
					initPos = 0;
					nlayerPos = nLs[0];
					scalePos = scaleL1;
				}
				else if (typePos == 1 || typePos == 2 || typePos == 3){
					initPos = nLs[0]*nLs[0];
					nlayerPos = nLs[1];	
					scalePos = scaleL23;				
				}
				else if (typePos == 4 || typePos == 5 || typePos == 6 || typePos == 7 || typePos == 8){
					initPos = nLs[0]*nLs[0] + nLs[1]*nLs[1];
					nlayerPos = nLs[2];	
					scalePos = scaleL4;				
				}
				else if (typePos == 9 || typePos == 10 || typePos == 11 || typePos == 12){
					initPos = nLs[0]*nLs[0] + nLs[1]*nLs[1] + nLs[2]*nLs[2];
					nlayerPos = nLs[3];	
					scalePos = scaleL5;				
				}
				else{
					initPos = nLs[0]*nLs[0] + nLs[1]*nLs[1] + nLs[2]*nLs[2] + nLs[3]*nLs[3];
					nlayerPos = nLs[4];		
					scalePos = scaleL6;		
				}			
				//define posicao (sem escala) do neuronio pos	
				xPos = (i - initPos)%nlayerPos;
				yPos = (i - initPos)/nlayerPos;	
				//varre lina do arquivo de conexao, coletando os neuronios pre-sinapticos de um determinado tipo no qual axonio atinge pos sinaptico
				for (j = 4; j < 21; j++){					
					if (connDat[p][j] > 0){
						//coleta informacao da arborizacao axonal do neuronio pre na camada informada pelo arquivo de conexao
						re = axonDat[j - 4][layerPos];
						//quanto neuronios pre de um detrminado tipo realizam sinapse neste neuronio pos
						numConn = (int) round(connDat[p][3]*(connDat[p][j]/(100.0*rescaleFac)));
						if (j == 4){
							beginK = 0;
							endK = nLs[0]*nLs[0];
						}
						else if (j == 5 || j == 6 || j == 7){
							beginK = nLs[0]*nLs[0];
							endK =  nLs[0]*nLs[0] + nLs[1]*nLs[1];				
						}
						else if (j == 8 || j == 9 || j == 10 || j == 11 || j == 12){
							beginK = nLs[0]*nLs[0] + nLs[1]*nLs[1];
							endK =  nLs[0]*nLs[0] + nLs[1]*nLs[1] + nLs[2]*nLs[2];		
						}
						else if (j == 13 || j == 14 || j == 15 || j == 16){
							beginK = nLs[0]*nLs[0] + nLs[1]*nLs[1] + nLs[2]*nLs[2];	
							endK = nLs[0]*nLs[0] + nLs[1]*nLs[1] + nLs[2]*nLs[2] + nLs[3]*nLs[3];		
						}
						else{
							beginK = nLs[0]*nLs[0] + nLs[1]*nLs[1] + nLs[2]*nLs[2] + nLs[3]*nLs[3];	
							endK = nneuron;	
						}		
						length = 0;				
						for (k = beginK; k < endK; k++){	
							typePre = typeList[k];	
							// define a camada, escala e offset na lista de neuronios respectivo ao neuronio pos em questao
							if(typePre == j - 4){
								if (typePre == 0){
									initPre = 0;
									nlayerPre = nLs[0];
									layerPre = 1;
									scalePre = scaleL1;
								}
								else if (typePre == 1 || typePre == 2 || typePre == 3){
									initPre = nLs[0]*nLs[0];
									nlayerPre = nLs[1];	
									layerPre = 2;
									scalePre = scaleL23;				
								}
								else if (typePre == 4 || typePre == 5 || typePre == 6 || typePre == 7 || typePre == 8){
									initPre = nLs[0]*nLs[0] + nLs[1]*nLs[1];
									nlayerPre = nLs[2];	
									layerPre = 3;	
									scalePre = scaleL4;			
								}
								else if (typePre == 9 || typePre == 10 || typePre == 11 || typePre == 12){
									initPre = nLs[0]*nLs[0] + nLs[1]*nLs[1] + nLs[2]*nLs[2];
									nlayerPre = nLs[3];		
									layerPre = 4;
									scalePre = scaleL5;			
								}
								else{
									initPre = nLs[0]*nLs[0] + nLs[1]*nLs[1] + nLs[2]*nLs[2] + nLs[3]*nLs[3];
									nlayerPre = nLs[4];		
									layerPre = 5;
									scalePre = scaleL6;		
								}	
								//define posicao (sem escala) do neuronio pre				
								xPre = (k - initPre)%nlayerPre;
								yPre = (k - initPre)/nlayerPre;	
								//verifica se neuronio pre atinge pos	
								dist = pow(xPre*scalePre-xPos*scalePos,2) + pow(yPre*scalePre-yPos*scalePos,2);
								if (dist < re*re){
									listNeuron[length] = k;
									length++;
								}
							}
						}
						
						//coletados neuronios potencialmente pre-sinapticos é realizado sorteio 
						for (k = 0; k < numConn; k++){		
							sort = round(next()*(length-1));					
							pre = listNeuron[(int) sort];
							
							MtConnAux = &MtConnColumn[i];					
							while(pre > MtConnAux->right->preSyn && MtConnAux->right->preSyn >= 0){
								MtConnAux = MtConnAux->right;
							}
							
							if(pre == MtConnAux->right->preSyn){
								//MtConnAux = MtConnAux->right;
								//MtConnAux->length++;
								//MtConnAux->weight = (float *) reallocc (MtConnAux->weight, MtConnAux->length*sizeof(float));
								//MtConnAux->weight[MtConnAux->length-1] = 0.5;						
								continue;
							}
							
							mtConn = (connect *) mallocc(sizeof(connect));
							mtConn->posSyn = i;
							mtConn->preSyn = pre;
							mtConn->length = 1;
							mtConn->weight = (float *) mallocc(mtConn->length*sizeof(float));
							mtConn->weight[0] = 1.0;
													
							mtConn->right = MtConnAux->right;
							MtConnAux->right = mtConn;
							MtConnColumn[i].length++;
							
							MtConnAux = &MtConnLine[pre];					
							while(i > MtConnAux->down->posSyn && MtConnAux->down->posSyn >= 0){
								MtConnAux = MtConnAux->down;
							}
							mtConn->down = MtConnAux->down;
							MtConnAux->down = mtConn;
							MtConnLine[pre].length++;				
						}
					}				
				}	
			}
		}	
	}
	return head;
}

connect * createRand (int nneuron, int numConn){
   	connect * mtConn, * MtConnAux, * head;
   	connect * MtConnLine, * MtConnColumn;
   	char flag = 1;
   	int k, pre, pos, i;
   	
    head = (connect *) mallocc(sizeof(connect));
	MtConnColumn = (connect *) mallocc(nneuron * sizeof(connect));
	MtConnLine = (connect *) mallocc(nneuron * sizeof(connect));
	
	head->posSyn = -1;
	head->preSyn = -1;
	head->weight = NULL;  
	head->down = &MtConnColumn[0];
	head->right = &MtConnLine[0];
	
    for (i = 0; i < nneuron-1; i++) {   	
		MtConnColumn[i].posSyn = i;
		MtConnColumn[i].preSyn = -1;
		MtConnColumn[i].weight = NULL;
		MtConnColumn[i].length = 0;
		MtConnColumn[i].right = &MtConnColumn[i];
		MtConnColumn[i].down = &MtConnColumn[i+1];	
		MtConnLine[i].posSyn = -1;
		MtConnLine[i].preSyn = i;
		MtConnLine[i].weight = NULL;
		MtConnLine[i].length = 0;
		MtConnLine[i].down = &MtConnLine[i];
		MtConnLine[i].right = &MtConnLine[i+1];  	
    }	
        
	MtConnColumn[nneuron-1].posSyn = nneuron-1;
	MtConnColumn[nneuron-1].preSyn = -1;
	MtConnColumn[nneuron-1].weight = NULL;
	MtConnColumn[nneuron-1].length = 0;
	MtConnColumn[nneuron-1].right = &MtConnColumn[nneuron-1];
	MtConnColumn[nneuron-1].down = &MtConnColumn[0];	
	MtConnLine[nneuron-1].posSyn = -1;
	MtConnLine[nneuron-1].preSyn = nneuron-1;
	MtConnLine[nneuron-1].weight = NULL;
	MtConnLine[nneuron-1].length = 0;
	MtConnLine[nneuron-1].down = &MtConnLine[nneuron-1];
	MtConnLine[nneuron-1].right = &MtConnLine[0];	
	
	for (k = 0; k < numConn; k++){		
		pre = (int) round(next()*(nneuron-1));
		pos = (int) round(next()*(nneuron-1));

		while (pos == pre)	
			pos = (int) round(next()*(nneuron-1));
	
		while (flag==0){	
			MtConnAux = MtConnLine[pre].down;
			while (MtConnAux->posSyn >= 0) {
				if (pos == MtConnAux->posSyn){
					pos = (int) round(next()*(nneuron-1));					
					while (pos == pre)	
						pos = (int) round(next()*(nneuron-1));										
					flag = 1;
					break;
				}
				MtConnAux = MtConnAux->down;
				flag = 0;
			}
		}
		
		MtConnAux = &MtConnLine[pre];					
		while(pos > MtConnAux->down->posSyn && MtConnAux->down->posSyn >= 0){
			MtConnAux = MtConnAux->down;
		}
		
		mtConn = (connect *) mallocc(sizeof(connect));
		mtConn->preSyn = pre;
		mtConn->posSyn = pos;
		mtConn->length = 1;
		mtConn->weight = (float *) mallocc(mtConn->length*sizeof(float));
		mtConn->weight[0] = 1.0;
								
		mtConn->down = MtConnAux->down;
		MtConnAux->down = mtConn;
		MtConnLine[pre].length++;
		
		MtConnAux = &MtConnColumn[pos];					
		while(pre > MtConnAux->right->preSyn && MtConnAux->right->preSyn >= 0){
			MtConnAux = MtConnAux->right;
		}
		mtConn->right = MtConnAux->right;
		MtConnAux->right = mtConn;
		MtConnColumn[pos].length++;				
	}	   	
	
	return head;	
}

connect * createRegNet (int nneuron, int numConn){
   	connect * mtConn, * MtConnAux, * head;
   	connect * MtConnLine, * MtConnColumn;
    
    int conNeuron = (int) round(numConn/nneuron);    
   	int k, i, pos;
    head = (connect *) mallocc(sizeof(connect));
	MtConnColumn = (connect *) mallocc(nneuron * sizeof(connect));
	MtConnLine = (connect *) mallocc(nneuron * sizeof(connect));
	
	head->posSyn = -1;
	head->preSyn = -1;
	head->weight = NULL;  
	head->down = &MtConnColumn[0];
	head->right = &MtConnLine[0];
	
    for (i = 0; i < nneuron-1; i++) {   	
		MtConnColumn[i].posSyn = i;
		MtConnColumn[i].preSyn = -1;
		MtConnColumn[i].weight = NULL;
		MtConnColumn[i].length = 0;
		MtConnColumn[i].right = &MtConnColumn[i];
		MtConnColumn[i].down = &MtConnColumn[i+1];	
		MtConnLine[i].posSyn = -1;
		MtConnLine[i].preSyn = i;
		MtConnLine[i].weight = NULL;
		MtConnLine[i].length = 0;
		MtConnLine[i].down = &MtConnLine[i];
		MtConnLine[i].right = &MtConnLine[i+1];  	
    }	
        
	MtConnColumn[nneuron-1].posSyn = nneuron-1;
	MtConnColumn[nneuron-1].preSyn = -1;
	MtConnColumn[nneuron-1].weight = NULL;
	MtConnColumn[nneuron-1].length = 0;
	MtConnColumn[nneuron-1].right = &MtConnColumn[nneuron-1];
	MtConnColumn[nneuron-1].down = &MtConnColumn[0];	
	MtConnLine[nneuron-1].posSyn = -1;
	MtConnLine[nneuron-1].preSyn = nneuron-1;
	MtConnLine[nneuron-1].weight = NULL;
	MtConnLine[nneuron-1].length = 0;
	MtConnLine[nneuron-1].down = &MtConnLine[nneuron-1];
	MtConnLine[nneuron-1].right = &MtConnLine[0];  
	
	for (k = 0; k < nneuron; k++){				
		for (i = 0; i < conNeuron; i++){				
			pos = (i - conNeuron/2) + k;			
			if (pos < 0){
				pos = nneuron + pos;
			}			
			if (pos > nneuron - 1){
				pos = pos - nneuron;
			}
			
			MtConnAux = &MtConnLine[k];					
			while(pos > MtConnAux->down->posSyn & MtConnAux->down->posSyn >=0){
				MtConnAux = MtConnAux->down;
			}
		
			mtConn = (connect *) mallocc(sizeof(connect));
			mtConn->preSyn = k;
			mtConn->posSyn = pos;
			mtConn->length = 1;
			mtConn->weight = (float *) mallocc(mtConn->length*sizeof(float));
			mtConn->weight[0] = 1.0;
									
			mtConn->down = MtConnAux->down;
			MtConnAux->down = mtConn;
			MtConnLine[k].length++;
			
			MtConnAux = &MtConnColumn[pos];					
			while(k > MtConnAux->right->preSyn && MtConnAux->right->preSyn >= 0){
				MtConnAux = MtConnAux->right;
			}
			mtConn->right = MtConnAux->right;
			MtConnAux->right = mtConn;
			MtConnColumn[pos].length++;			
		}		
	}	   	
	
	return head;	
}

connect * createHMNet (int nneuron, int m, float prob, float probEx){
   	
    int i, j, count, sizem = nneuron, num, mod1, mod2, n1, n2, nn;
    double sort;
    
    int ** listsyn = (int **) mallocc( 1000000 * sizeof (int *));   	
	for (i=0; i<1000000; i++)
		listsyn[i] = (int *) mallocc( 2 * sizeof (int));		
    
    short int * typeCell = (short int *) mallocc(nneuron * sizeof(short int));
    
    for (i = 0; i < nneuron; i++) {      
        num = (int) (next()*5);
        if (num == 4){
			typeCell[i] = 0;
        }
        else{
			typeCell[i] = 1;
        }       
    }
    
    count = 0;
	for (i = 0; i < nneuron; i++) {  
		for (j = 0; j < nneuron; j++) {
			sort = next();			
			if ((j!=i) & (sort<prob)) {
				listsyn[count][0] = i;
				listsyn[count][1] = j;	
				count++;
			}
		}	
	}
	listsyn[count][0] = -1;	
	
    for (i = 1; i <= m; i++){
    	sizem = sizem/2;
    	count = 0;
		while(listsyn[count][0] != -1){  									
			n1 = 1; n2 = 1; mod1 = 0; mod2 = 0;			
			while(listsyn[count][1] >= mod2 + sizem){
				mod2 += sizem;
				n2++;	
			}												
			while(listsyn[count][0] >= mod1 + sizem){
				mod1 += sizem;
				n1++;	
			}				
			if ((n1!=n2) & (typeCell[listsyn[count][0]] == 0)){	
				nn = mod1 + (int) (next()*sizem);
				listsyn[count][1] = nn;				
			}
			if ((n1!=n2) & (typeCell[listsyn[count][0]] == 1)){	
				sort = next();
				if (sort < probEx) {
					nn = mod1 + (int) (next()*sizem);
					listsyn[count][1] = nn;	
				}										
			}							  	
	    	count++;
    	}
    }    
   	
   	connect * mtConn, * MtConnAux, * head;
   	connect * MtConnLine, * MtConnColumn;   
    head = (connect *) mallocc(sizeof(connect));
	MtConnColumn = (connect *) mallocc(nneuron * sizeof(connect));
	MtConnLine = (connect *) mallocc(nneuron * sizeof(connect));
	
	head->posSyn = -1;
	head->preSyn = -1;
	head->weight = NULL;  
	head->down = &MtConnColumn[0];
	head->right = &MtConnLine[0];
	
    for (i = 0; i < nneuron-1; i++) {   	
		MtConnColumn[i].posSyn = i;
		MtConnColumn[i].preSyn = -1;
		MtConnColumn[i].weight = NULL;
		MtConnColumn[i].length = 0;
		MtConnColumn[i].right = &MtConnColumn[i];
		MtConnColumn[i].down = &MtConnColumn[i+1];	
		MtConnLine[i].posSyn = -1;
		MtConnLine[i].preSyn = i;
		MtConnLine[i].weight = NULL;
		MtConnLine[i].length = 0;
		MtConnLine[i].down = &MtConnLine[i];
		MtConnLine[i].right = &MtConnLine[i+1];  	
    }	
        
	MtConnColumn[nneuron-1].posSyn = nneuron-1;
	MtConnColumn[nneuron-1].preSyn = -1;
	MtConnColumn[nneuron-1].weight = NULL;
	MtConnColumn[nneuron-1].length = 0;
	MtConnColumn[nneuron-1].right = &MtConnColumn[nneuron-1];
	MtConnColumn[nneuron-1].down = &MtConnColumn[0];	
	MtConnLine[nneuron-1].posSyn = -1;
	MtConnLine[nneuron-1].preSyn = nneuron-1;
	MtConnLine[nneuron-1].weight = NULL;
	MtConnLine[nneuron-1].length = 0;
	MtConnLine[nneuron-1].down = &MtConnLine[nneuron-1];
	MtConnLine[nneuron-1].right = &MtConnLine[0];  
	
	int pre, pos;
	count = 0;
	while(listsyn[count][0] != -1){
		pre = listsyn[count][0];
		pos = listsyn[count][1];
				
		MtConnAux = &MtConnLine[pre];					
		while(pos > MtConnAux->down->posSyn & MtConnAux->down->posSyn >=0){
			MtConnAux = MtConnAux->down;
		}
		
		if(pos == MtConnAux->down->posSyn){
			MtConnAux = MtConnAux->down;
			MtConnAux->length++;
			count++;
			continue;
		}	
		
		mtConn = (connect *) mallocc(sizeof(connect));
		mtConn->preSyn = pre;
		mtConn->posSyn = pos;
		mtConn->length = 1;
		mtConn->weight = (float *) mallocc(mtConn->length*sizeof(float));
		mtConn->weight[0] = 1.0;
								
		mtConn->down = MtConnAux->down;
		MtConnAux->down = mtConn;
		MtConnLine[pre].length++;
		
		MtConnAux = MtConnColumn[pos].right;					
		while(pre > MtConnAux->right->preSyn && MtConnAux->right->preSyn >= 0){
			MtConnAux = MtConnAux->right;
		}
		mtConn->right = MtConnAux->right;
		MtConnAux->right = mtConn;
		MtConnColumn[pos].length++;
		count++;		
	}
	
	return head;	
}

void desalocar(connect * head, int nneuron){
	connect * mtxPre, * mtxPos, * mtx, * mtxAux;
	mtxPre = head->right;
	mtxPos = head->down;

	int i;
	for(i = 0; i < nneuron; i++){
		mtxAux = mtxPre[i].down;
		while (mtxAux->posSyn >= 0){
			mtx = mtxAux;
			mtxAux = mtx->down;
			free(mtx);	
		}
	}
	
	free(head->right);
	free(head->down);
	free(head);	
}

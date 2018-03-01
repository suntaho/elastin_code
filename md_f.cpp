#include "mex.h" 
#include "matrix.h"
#include "stdlib.h"
#include "stdio.h"
#include "math.h"
#include "time.h"
//#include "conio.h"
#include "omp.h"



// define global variables for molecular dynamics (MD),
// setting kb=1, mass=1.
double nstep = 10000;					    // total steps for MD
double dt = 0.001;							// time interval
double delta = 0.001;
double tauT = 10.0*dt;						// rise time of Berendsen Thermostat
int neiggrid = 1;							// the cut-off of L-J by grid number
double dampcoeff = 1.0;					    // damping force ~ -dampcoeff*velocity
int itp_method = 1;                         // interpolation method 0: trilinear, 1: tricubic
int defect_thd_n = 4;						// define a defect as occupied grids <= defect_thd 
double lmax_ini,lmax,lmax_stable;			// maximum lenght for bonds to avoid atom escape 
// define structure variables
struct matlabData{
	int dimx, dimy, dimz, ncpu;
	double ***img3d, ***img3dg;
	double *img3do, *imgxavg, *imgyavg, *imgzavg;
};
struct mdrun{
	int dimx, dimy, dimz, ncpu;
	long natm, group, nbond;
	long **bondij, **bondij_bgn_end;
	long ***mapgrid, *numlist, **ilist;
	long ***mapdefect;
	double sig, eps, T, k;
	double *fx, *fy, *fz;
	double *vx, *vy, *vz;
	double  *x,  *y,  *z;
	double *fxnew, *fynew, *fznew;
	double *vxnew, *vynew, *vznew;
	double  *xnew,  *ynew,  *znew;
	double  *x0,  *y0,  *z0;
	double ***xavg, ***yavg, ***zavg;
	double ****intp3;
	double tblinv[64][64];
};



// define subroutine function	
double interp(double xl, double yl, double zl, double v[2][2][2], double vc[64], int method);	// method 0: trilinear, 1: tricubic
void freeMem(struct matlabData *imgs, struct mdrun *md);			// free memory
void md_cal(struct matlabData *imgs, struct mdrun *md);				// run md
void vel_rescale(struct mdrun *md, double tt);						// rescale velocity to meet assigned temperature
void gen_neighbor_list(struct mdrun *md);							// generate list of neighbor atoms for L-J potential
void gen_bond_list(struct mdrun *md);								// generate list of bonding atoms
void force_cal(struct matlabData *imgs, struct mdrun *md);          // calculate force on atomi;
void gen_intp3(struct matlabData *imgs, struct mdrun *md);          // generate table for tricubic interpolation
void export_xyz(struct mdrun *md, char fi);                         // output xyz
void map_defect(struct mdrun *md);									// map defect
void map_defect_add_atom(struct mdrun *md);							// map defect with inserting extra atoms along too-long bond


void mexFunction( 
	int nlhs, mxArray *plhs[], 
	int nrhs, const mxArray *prhs[]) 
{ 
	int i, j, k;
	double *rbuf, *rbuf1, *rbuf2;
	// matlab variables;
	const mwSize *dims;
	// struct variables
	struct matlabData imgs;
	struct mdrun md;
	

	// obtain dimension of input array
	dims = mxGetDimensions(prhs[0]);
	imgs.dimx = int(*(dims)); imgs.dimy = int(*(dims+1)); imgs.dimz = int(*(dims+2));
	md.dimx = int(*(dims));   md.dimy = int(*(dims+1));   md.dimz = int(*(dims+2)); 
	
	// generate array for variables by C++
	imgs.img3d = new double**[imgs.dimx];
	for (i=0; i<imgs.dimx; i++){
	  imgs.img3d[i] = new double*[imgs.dimy];
	  for (j=0; j<imgs.dimy; j++){ imgs.img3d[i][j] = new double[imgs.dimz]; }
	}
	imgs.img3dg = new double**[imgs.dimx];
	for (i=0; i<imgs.dimx; i++){
	  imgs.img3dg[i] = new double*[imgs.dimy];
	  for (j=0; j<imgs.dimy; j++){ imgs.img3dg[i][j] = new double[imgs.dimz]; }
	}
	md.mapgrid = new long**[imgs.dimx];
	for (i=0; i<imgs.dimx; i++){
	  md.mapgrid[i] = new long*[imgs.dimy];
	  for (j=0; j<imgs.dimy; j++){ md.mapgrid[i][j] = new long[imgs.dimz]; }
	}
	md.mapdefect = new long**[imgs.dimx];
	for (i=0; i<imgs.dimx; i++){
	  md.mapdefect[i] = new long*[imgs.dimy];
	  for (j=0; j<imgs.dimy; j++){ md.mapdefect[i][j] = new long[imgs.dimz]; }
	}
	md.xavg = new double**[imgs.dimx];
	for (i=0; i<imgs.dimx; i++){
	  md.xavg[i] = new double*[imgs.dimy];
	  for (j=0; j<imgs.dimy; j++){ md.xavg[i][j] = new double[imgs.dimz]; }
	}
	md.yavg = new double**[imgs.dimx];
	for (i=0; i<imgs.dimx; i++){
	  md.yavg[i] = new double*[imgs.dimy];
	  for (j=0; j<imgs.dimy; j++){ md.yavg[i][j] = new double[imgs.dimz]; }
	}
	md.zavg = new double**[imgs.dimx];
	for (i=0; i<imgs.dimx; i++){
	  md.zavg[i] = new double*[imgs.dimy];
	  for (j=0; j<imgs.dimy; j++){ md.zavg[i][j] = new double[imgs.dimz]; }
	}
	
	// generate array for output img3d
	plhs[0] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
	plhs[1] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
	plhs[2] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
	plhs[3] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
	
	// get data from input
	rbuf = mxGetPr(prhs[2]);  imgs.ncpu = int(*rbuf); md.ncpu = int(*rbuf);
	rbuf = mxGetPr(prhs[3]);  md.eps = *rbuf;
	rbuf = mxGetPr(prhs[4]);  md.sig = *rbuf;
	rbuf = mxGetPr(prhs[5]);  md.T = *rbuf;
	rbuf = mxGetPr(prhs[6]);  md.k = *rbuf;
	rbuf1 = mxGetPr(prhs[0]); rbuf2 = mxGetPr(prhs[1]);
	md.natm = 0; md.nbond = 0;
	for(i=0; i<imgs.dimx; i++){
		for(j=0; j<imgs.dimy; j++){
			for(k=0; k<imgs.dimz; k++){
				imgs.img3d[i][j][k] =* ( rbuf1+k*(imgs.dimy*imgs.dimx)+j*imgs.dimx+i );  // note the sorting order in matlab is z->y->x
				imgs.img3dg[i][j][k] =* ( rbuf2+k*(imgs.dimy*imgs.dimx)+j*imgs.dimx+i );		 
				md.natm = md.natm+long(imgs.img3d[i][j][k]);
	}	} 	}
	md.natm=2*md.natm;
	
	// initialize md variables
	md.fx = new double[md.natm]; md.fy = new double[md.natm]; md.fz = new double[md.natm];
	md.vx = new double[md.natm]; md.vy = new double[md.natm]; md.vz = new double[md.natm];
	md.x = new double[md.natm];  md.y = new double[md.natm];  md.z = new double[md.natm];
	md.x0 = new double[md.natm];  md.y0 = new double[md.natm];  md.z0 = new double[md.natm];
	md.fxnew = new double[md.natm]; md.fynew = new double[md.natm]; md.fznew = new double[md.natm];
	md.vxnew = new double[md.natm]; md.vynew = new double[md.natm]; md.vznew = new double[md.natm];
	md.xnew = new double[md.natm];  md.ynew = new double[md.natm];  md.znew = new double[md.natm];
	md.group = 1; md.numlist = new long[1];
	md.ilist = new long*[1]; md.ilist[0]=new long[1];

	// initialized table for tricubic interpolation
	if(itp_method==1) {
		md.intp3 = new double***[imgs.dimx];
		for (i=0; i<imgs.dimx; i++){
		  md.intp3[i] = new double**[imgs.dimy];
		  for (j=0; j<imgs.dimy; j++){ 
			md.intp3[i][j] = new double*[imgs.dimz]; 
			for (k=0; k<imgs.dimz; k++) { md.intp3[i][j][k] = new double[64]; }
		  }
		}
		gen_intp3(&imgs, &md);
	}
	
	// cal. MD
	md_cal(&imgs, &md);
	
	// output img3d
	imgs.img3do = mxGetPr(plhs[0]);
	for(i=0; i<imgs.dimx; i++){	
		for(j=0; j<imgs.dimy; j++){
			for(k=0; k<imgs.dimz; k++){
				if(md.mapdefect[i][j][k]<0) { *( imgs.img3do+k*(imgs.dimy*imgs.dimx)+j*imgs.dimx+i ) = 0; }
				else{ *( imgs.img3do+k*(imgs.dimy*imgs.dimx)+j*imgs.dimx+i ) = 1; }
	}	}	}
	imgs.imgxavg = mxGetPr(plhs[1]);
	for(i=0; i<imgs.dimx; i++){	
		for(j=0; j<imgs.dimy; j++){
			for(k=0; k<imgs.dimz; k++){
				*( imgs.imgxavg+k*(imgs.dimy*imgs.dimx)+j*imgs.dimx+i ) = md.xavg[i][j][k];
	}	}	}
	imgs.imgyavg = mxGetPr(plhs[2]);
	for(i=0; i<imgs.dimx; i++){	
		for(j=0; j<imgs.dimy; j++){
			for(k=0; k<imgs.dimz; k++){
				*( imgs.imgyavg+k*(imgs.dimy*imgs.dimx)+j*imgs.dimx+i ) = md.yavg[i][j][k];
	}	}	}
	imgs.imgzavg = mxGetPr(plhs[3]);
	for(i=0; i<imgs.dimx; i++){	
		for(j=0; j<imgs.dimy; j++){
			for(k=0; k<imgs.dimz; k++){
				*( imgs.imgzavg+k*(imgs.dimy*imgs.dimx)+j*imgs.dimx+i ) = md.zavg[i][j][k];
	}	}	}
	
	// free memory
	freeMem(&imgs, &md);
} 



// free array memory
void freeMem(struct matlabData *imgs, struct mdrun *md){
	
	if(itp_method==1) {
		// 4dim variablesfor (i=0; i<imgs.dimx; i++){
		for (int i=0; i<imgs->dimx; i++){
		  for (int j=0; j<imgs->dimy; j++){  
			for (int k=0; k<imgs->dimz; k++) { delete [] md->intp3[i][j][k]; }
			delete [] md->intp3[i][j];
		  }
		  delete [] md->intp3[i];
		}
		delete md->intp3;
	}
	
	// 3dim variables
	for (int i=0; i<imgs->dimx; i++){
		for (int j=0; j<imgs->dimy; j++){
			delete [] imgs->img3d[i][j];
			delete [] imgs->img3dg[i][j];
			delete [] md->mapgrid[i][j];
			delete [] md->mapdefect[i][j];
			delete [] md->xavg[i][j];
			delete [] md->yavg[i][j];
			delete [] md->zavg[i][j];
		}
		delete [] imgs->img3d[i];
		delete [] imgs->img3dg[i];
		delete [] md->mapgrid[i];
		delete [] md->mapdefect[i];
		delete [] md->xavg[i];
		delete [] md->yavg[i];
		delete [] md->zavg[i];
	}
	delete [] imgs->img3d;
	delete [] imgs->img3dg;
	delete [] md->mapgrid;
	delete [] md->mapdefect;
	delete [] md->xavg;
	delete [] md->yavg;
	delete [] md->zavg;

	// 2dim variables
	for(int i=0; i<md->group; i++){ delete [] md->ilist[i]; }
	delete [] md->ilist;
	for(int i=0; i<md->nbond; i++){ delete [] md->bondij[i]; }
	delete [] md->bondij;
	for(int i=0; i<md->natm; i++){ delete [] md->bondij_bgn_end[i]; }
	delete [] md->bondij_bgn_end;
	
	// 1dim variables
	delete [] md->fx; delete [] md->fy; delete [] md->fz;
	delete [] md->vx; delete [] md->vy; delete [] md->vz;
	delete [] md->x;  delete [] md->y;  delete [] md->z;
	delete [] md->x0;  delete [] md->y0;  delete [] md->z0;
	delete [] md->fxnew; delete [] md->fynew; delete [] md->fznew;
	delete [] md->vxnew; delete [] md->vynew; delete [] md->vznew;
	delete [] md->xnew;  delete [] md->ynew;  delete [] md->znew;
	delete [] md->numlist;
}



// generate table for tricubic interpolation
void gen_intp3(struct matlabData *imgs, struct mdrun *md){
	int i, j, k, u, v, w, ui, vi, wi, n;
	int idxi, idxj;
	double tm1, tm2, tm3, ei;
	double tbl[64][64], tblbuf[64][64], vptn[64];
	
	
	// build coefficient matrix
	idxi = 0;
	for(i=-1; i<=2; i++){ for(j=-1; j<=2; j++){ for(k=-1; k<=2; k++){
		idxj = 0;
		for(u=0; u<=3; u++){ 
			if( i==0 && u==0) { tm1=1.0; } else { tm1 = 1.0*pow(1.0*i,1.0*u); }
			for(v=0; v<=3; v++){ 
				if( j==0 && v==0) { tm2=1.0; } else { tm2 = 1.0*pow(1.0*j,1.0*v); }
				for(w=0; w<=3; w++){
					if( k==0 && w==0) { tm3=1.0; } else { tm3 = 1.0*pow(1.0*k,1.0*w); }
					tbl[idxi][idxj] = tm1*tm2*tm3;   idxj = idxj+1;
		} } }
		idxi = idxi+1;
	} } }

    // initialize inverse matrix 
	for (i=0; i<64; i++){ for(j=0; j<64; j++) {			//initialize as unitary matrix
			if(i==j){ md->tblinv[i][j]=1.0; } else { md->tblinv[i][j]=0.0; }
			tblbuf[i][j]=tbl[i][j];
	}   }
	for (i=0; i<64-1; i++){ for(j=i+1; j<64; j++) {     // cal upper matrix
			ei=tblbuf[j][i]/tblbuf[i][i];
			for(k=i; k<64; k++){ tblbuf[j][k] = tblbuf[j][k]-tblbuf[i][k]*ei;}
			for(k=0; k<64; k++){ md->tblinv[j][k] = md->tblinv[j][k]-md->tblinv[i][k]*ei;}
    }   }
	for (i=64-1; i>0; i--){ for(j=i-1; j>=0; j--) {     // cal lower matrix
			ei=tblbuf[j][i]/tblbuf[i][i];
			for(k=0; k<64; k++){ tblbuf[j][k] = tblbuf[j][k]-tblbuf[i][k]*ei;}
			for(k=0; k<64; k++){ md->tblinv[j][k] = md->tblinv[j][k]-md->tblinv[i][k]*ei;}
	}   }
	for (i=0; i<64; i++){ for(j=0; j<64; j++){          // obtain inverse matrix
			md->tblinv[i][j] = md->tblinv[i][j]/tblbuf[i][i];
	}   }
	
	// build table for tricubic interpolation
	#pragma omp parallel for private(i,j,k,idxi,u,v,w,ui,vi,wi,vptn,n) num_threads(md->ncpu)
	for( i=0; i<imgs->dimx; i++){ for( j=0; j<imgs->dimy; j++){ for( k=0; k<imgs->dimz; k++){
		idxi = 0;
		for( u=i-1; u<=i+2; u++ ){ for( v=j-1; v<=j+2; v++){ for( w=k-1; w<=k+2; w++){
			ui = u; vi = v; wi = w;
			if( ui<0 ){ ui = ui+imgs->dimx; } else if( ui>=imgs->dimx ){ ui = ui-imgs->dimx; }
			if( vi<0 ){ vi = vi+imgs->dimy; } else if( vi>=imgs->dimy ){ vi = vi-imgs->dimy; }
			if( wi<0 ){ wi = wi+imgs->dimz; } else if( wi>=imgs->dimz ){ wi = wi-imgs->dimz; }
			vptn[idxi] = imgs->img3dg[ui][vi][wi]; idxi = idxi+1;
		} } }
		for( n=0; n<64; n++ ){
			md->intp3[i][j][k][n] = 0.0;
			for( u=0; u<64; u++ ){
				md->intp3[i][j][k][n] = md->intp3[i][j][k][n]+md->tblinv[n][u]*vptn[u];
		}  }
	} } }
	#pragma omp barrier	
	
}



// interpolation by method 0: trilinear, 1: tricubic
double interp(double xl, double yl, double zl, double v[2][2][2], double vc[64], int method){
	int i, j, k, idx;
	double tm1, tm2, tm3;
	double c00, c01, c10, c11;
	double c0, c1, c;

	
	if( method ==0 ){
		// trilinear interpolation
		c00 = v[0][0][0]*(1.0-xl)+v[1][0][0]*xl;
		c01 = v[0][0][1]*(1.0-xl)+v[1][0][1]*xl;
		c10 = v[0][1][0]*(1.0-xl)+v[1][1][0]*xl;
		c11 = v[0][1][1]*(1.0-xl)+v[1][1][1]*xl;
		c0 = c00*(1.0-yl)+c10*yl;
		c1 = c01*(1.0-yl)+c11*yl;
		c = c0*(1.0-zl)+c1*zl;
	}
	else if( method ==1 ){
		// tricubic interpolation
		c = 0.0; idx = 0;
		for( i=0; i<=3; i++ ){
			if (i==0) { tm1=1.0;} else if(i==1){tm1=xl;} else if(i==2){tm1=xl*xl;} else if(i==3){tm1=xl*xl*xl;} 
			for( j=0; j<=3; j++ ){
				if (j==0) { tm2=1.0;} else if(j==1){tm2=yl;} else if(j==2){tm2=yl*yl;} else if(j==3){tm2=yl*yl*yl;} 
				for( k=0; k<=3; k++ ){
					if (k==0) { tm3=1.0;} else if(k==1){tm3=zl;} else if(k==2){tm3=zl*zl;} else if(k==3){tm3=zl*zl*zl;} 
					c = c+vc[idx]*tm1*tm2*tm3; idx = idx+1;
		}  }  }
	}
	else{ printf("error on setting interpolation method!\n"); c=0.0; }
	
	return c;
}



// rescale velocity to meet assign temperature
void vel_rescale(struct mdrun *md, double tt){
	int i;
	double eng = 0.0, lambda;
	
	
	for(i=0; i<md->natm; i++){
		eng = eng+ md->vx[i]*md->vx[i] + md->vy[i]*md->vy[i] + md->vz[i]*md->vz[i] ;
	}
	eng = eng/(3.0*md->natm);
	if(eng<delta) {eng=delta;}
	lambda = sqrt(1.0+tt*(md->T/eng-1.0)/tauT);
	for(i=0; i<md->natm; i++){
		md->vx[i]=lambda*md->vx[i]; md->vy[i]=lambda*md->vy[i]; md->vz[i]=lambda*md->vz[i];
	}
}



// generate list of bonding atoms
void gen_bond_list(struct mdrun *md){
	long i, listindex, atmindex, atmcount, idx;
	int in, jn, kn;
	int ic, jc ,kc, ii, jj, kk;
	double shfx, shfy, shfz;
	
	
	// calculate number of bonds
	idx = 0;
	for(i=0;i < md->natm;i++){
		ic = int(floor(md->xnew[i])); jc = int(floor(md->ynew[i])); kc = int(floor(md->znew[i]));
		for(ii=ic-neiggrid; ii<=(ic+neiggrid); ii++){ for(jj=jc-neiggrid; jj<=(jc+neiggrid); jj++){ for(kk=kc-neiggrid ;kk<=(kc+neiggrid); kk++) {
			in = ii; jn = jj; kn = kk; shfx=0.0; shfy=0.0; shfz=0.0;
			if(in<0)                { in=in+md->dimx; shfx=-md->dimx; }
			else if(in >= md->dimx) { in=in-md->dimx; shfx= md->dimx; }
			if(jn<0)                { jn=jn+md->dimy; shfy=-md->dimy; }
			else if(jn >= md->dimy) { jn=jn-md->dimy; shfy= md->dimy; }
			if(kn<0)                { kn=kn+md->dimz; shfz=-md->dimz; }
			else if(kn >= md->dimz) { kn=kn-md->dimz; shfz= md->dimz; }
			listindex = md->mapgrid[in][jn][kn];
			if(listindex>=0){ idx = idx+md->numlist[listindex]; }
		}}}		
	}
	md->nbond = idx;
	
	// initialized array
	md->bondij = new long*[md->nbond];
	for (i=0; i<md->nbond; i++){ md->bondij[i] = new long[2];}
	md->bondij_bgn_end = new long*[md->natm];
	for (i=0; i<md->natm; i++){ md->bondij_bgn_end[i] = new long[2];}
	
	// build bonding list
	idx = 0;
	for(i=0;i < md->natm;i++){
		md->bondij_bgn_end[i][0] = idx;   // the beginning index of md->bondij for atom i
		ic = int(floor(md->xnew[i])); jc = int(floor(md->ynew[i])); kc = int(floor(md->znew[i]));
		for(ii=ic-neiggrid; ii<=(ic+neiggrid); ii++){ for(jj=jc-neiggrid; jj<=(jc+neiggrid); jj++){ for(kk=kc-neiggrid ;kk<=(kc+neiggrid); kk++) {
			in = ii; jn = jj; kn = kk; shfx=0.0; shfy=0.0; shfz=0.0;
			if(in<0)                { in=in+md->dimx; shfx=-md->dimx; }
			else if(in >= md->dimx) { in=in-md->dimx; shfx= md->dimx; }
			if(jn<0)                { jn=jn+md->dimy; shfy=-md->dimy; }
			else if(jn >= md->dimy) { jn=jn-md->dimy; shfy= md->dimy; }
			if(kn<0)                { kn=kn+md->dimz; shfz=-md->dimz; }
			else if(kn >= md->dimz) { kn=kn-md->dimz; shfz= md->dimz; }
			listindex = md->mapgrid[in][jn][kn];
			if(listindex>=0){
				for(atmcount=1; atmcount <= md->numlist[listindex]; atmcount++){
					atmindex = md->ilist[listindex][atmcount-1];
					md->bondij[idx][0] = i; md->bondij[idx][1] = atmindex; idx = idx+1; 
				}
			}
		}}}		
		md->bondij_bgn_end[i][1] = idx-1;   // the ending index of md->bondij for atom i
	}
	if( idx != md->nbond){ printf("Warning! error on generation of bonding list!\n"); }
}



// generate list of neighbor atoms for L-J potential
void gen_neighbor_list(struct mdrun *md){
	int i, j, k, nmax=0;
	long ii, groupbuf, indexi, indexj;
	

	// initialize array
	#pragma omp parallel for private(i,j,k) num_threads(md->ncpu)
	for(i=0 ; i<md->dimx; i++){
		for(j=0 ; j<md->dimy; j++){
			for(k=0 ; k<md->dimz; k++){
				md->mapgrid[i][j][k]=0;
	}	}	}
	#pragma omp barrier	
	
	// find the maximum of atom in single grid
	groupbuf = 0;
	for(ii=0; ii<md->natm; ii++){
		i =int(floor(md->xnew[ii])); j =int(floor(md->ynew[ii])); k =int(floor(md->znew[ii]));
		md->mapgrid[i][j][k] = md->mapgrid[i][j][k]+1;
		if(md->mapgrid[i][j][k]>nmax){ nmax=md->mapgrid[i][j][k]; }
		if(md->mapgrid[i][j][k]==1){groupbuf=groupbuf+1;}
	}
	
	// rebuild array for neighbor list
	for(i=0; i<md->group; i++){ delete [] md->ilist[i]; }
	delete [] md->ilist; delete [] md->numlist;
	md->group = groupbuf;
	md->numlist = new long[md->group]; md->ilist = new long*[md->group];
	for(i=0; i<md->group; i++){ md->ilist[i] = new long[nmax]; }
	for(i=0; i<md->group; i++){ md->numlist[i]=0; }				// initialize array
	for(i=0; i<md->group; i++){ for(j=0; j<nmax; j++) {md->ilist[i][j]=0; } }
	
	// assign value to list
	ii = 0;
	for(i=0 ; i<md->dimx; i++){
		for(j=0 ; j<md->dimy; j++){
			for(k=0 ; k<md->dimz; k++){
				if(md->mapgrid[i][j][k]>0){
					md->mapgrid[i][j][k] = ii;   ii = ii+1;   // change to map index between 3D grid and 1D list
				}
				else {md->mapgrid[i][j][k] = -1;}
	}	}	}
	for(ii=0; ii<md->natm; ii++){
		i =int(floor(md->xnew[ii])); j =int(floor(md->ynew[ii])); k =int(floor(md->znew[ii]));
		indexi = md->mapgrid[i][j][k]; indexj = md->numlist[indexi];
		md->numlist[indexi] = md->numlist[indexi]+1;
		md->ilist[indexi][indexj] = ii;
	}
	
}



// filter defect
void map_defect(struct mdrun *md){
	int i, j, k, u, v, w, ui, vi, wi, idx1, idx2;
	
	
	// initialized matrix for filter defect
	for(i=0 ; i<md->dimx; i++){
		for(j=0 ; j<md->dimy; j++){
			for(k=0 ; k<md->dimz; k++){
				md->mapdefect[i][j][k] = md->mapgrid[i][j][k];
	}	}	}
	
	// filter defects
	for( i=0; i<md->dimx; i++){ for( j=0; j<md->dimy; j++){ for( k=0; k<md->dimz; k++){
		idx1 = 0; idx2 = 0; 
		for( u=i-1; u<=i+1; u++ ){ for( v=j-1; v<=j+1; v++){ for( w=k-1; w<=k+1; w++){
			ui = u; vi = v; wi = w;
			if( ui<0 ){ ui = ui+md->dimx; } else if( ui>=md->dimx ){ ui = ui-md->dimx; }
			if( vi<0 ){ vi = vi+md->dimy; } else if( vi>=md->dimy ){ vi = vi-md->dimy; }
			if( wi<0 ){ wi = wi+md->dimz; } else if( wi>=md->dimz ){ wi = wi-md->dimz; }
			if( md->mapdefect[ui][vi][wi]>=0 ){ idx1 = idx1+1; }
		} } }
		for( u=i-3; u<=i+3; u++ ){ for( v=j-3; v<=j+3; v++){ for( w=k-3; w<=k+3; w++){
			ui = u; vi = v; wi = w;
			if( ui<0 ){ ui = ui+md->dimx; } else if( ui>=md->dimx ){ ui = ui-md->dimx; }
			if( vi<0 ){ vi = vi+md->dimy; } else if( vi>=md->dimy ){ vi = vi-md->dimy; }
			if( wi<0 ){ wi = wi+md->dimz; } else if( wi>=md->dimz ){ wi = wi-md->dimz; }
			if( md->mapdefect[ui][vi][wi]>=0 ){ idx2 = idx2+1; }
		} } }
		if( idx1==idx2 && idx1<=defect_thd_n ){
			for( u=i-1; u<=i+1; u++ ){ for( v=j-1; v<=j+1; v++){ for( w=k-1; w<=k+1; w++){
				ui = u; vi = v; wi = w;
				if( ui<0 ){ ui = ui+md->dimx; } else if( ui>=md->dimx ){ ui = ui-md->dimx; }
				if( vi<0 ){ vi = vi+md->dimy; } else if( vi>=md->dimy ){ vi = vi-md->dimy; }
				if( wi<0 ){ wi = wi+md->dimz; } else if( wi>=md->dimz ){ wi = wi-md->dimz; }
				md->mapdefect[ui][vi][wi]=-1;
			} } }
		}
	} } }
}



// filter defect with inserting extra atoms along too-long bond 
void map_defect_add_atom(struct mdrun *md){
	long ii, atm1, atm2, ***navg;
	int i, j, k, u, v, w, ui, vi, wi, idx1, idx2, ninsert;
	int in, jn, kn;
	double db1, db2, db3, xi, yi, zi, r1;
	
	
	// initialize array memory
	navg = new long**[md->dimx];
	for (i=0; i<md->dimx; i++){
	  navg[i] = new long*[md->dimy];
	  for (j=0; j<md->dimy; j++){ navg[i][j] = new long[md->dimz]; }
	}
	
	// initialized matrix for filter defect
	for(i=0 ; i<md->dimx; i++){
		for(j=0 ; j<md->dimy; j++){
			for(k=0 ; k<md->dimz; k++){
				if (md->mapgrid[i][j][k]>=0) { md->mapdefect[i][j][k] = 1;}
				else { md->mapdefect[i][j][k] = -1; }
				md->xavg[i][j][k] = 0.0; md->yavg[i][j][k] = 0.0; md->zavg[i][j][k] = 0.0;
				navg[i][j][k] = 0;
	}	}	}
	
	// insert extra atoms
	for(ii=0; ii<md->natm; ii++){
		for(j=md->bondij_bgn_end[ii][0]; j<=md->bondij_bgn_end[ii][1]; j++){
			atm1 = md->bondij[j][0]; atm2 = md->bondij[j][1];
			if( atm1 < atm2 ){
				db1 = md->xnew[atm2]-md->xnew[atm1]; 
				db2 = md->ynew[atm2]-md->ynew[atm1];
				db3 = md->znew[atm2]-md->znew[atm1];
				r1 = sqrt(db1*db1+db2*db2+db3*db3);
				if( r1==0.0 ){ r1=printf("r1 is zero at atoms %d and %d\n",atm1,atm2); }
				if( r1>0.5*md->dimx || r1>0.5*md->dimy || r1>0.5*md->dimz){ printf("Inserting atoms along bonds error!!\n"); }
				else{  // ignore bonds between atoms across periodic boundary
					ninsert = int(3.0*floor(r1));
					for( k=0; k<=ninsert+1; k++ ){
						xi = md->xnew[atm1]+db1*k/(1.0+ninsert);
						yi = md->ynew[atm1]+db2*k/(1.0+ninsert);
						zi = md->znew[atm1]+db3*k/(1.0+ninsert);
						in =int(floor(xi)); jn =int(floor(yi)); kn =int(floor(zi));
						md->mapdefect[in][jn][kn] = 1;
						md->xavg[in][jn][kn] = md->xavg[in][jn][kn]+xi; 
						md->yavg[in][jn][kn] = md->yavg[in][jn][kn]+yi; 
						md->zavg[in][jn][kn] = md->zavg[in][jn][kn]+zi;
						navg[in][jn][kn] = navg[in][jn][kn]+1;
					}
				}
			}
		}
	}
	
	// calculate xavg, yavg, zavg
	for(i=0 ; i<md->dimx; i++){
		for(j=0 ; j<md->dimy; j++){
			for(k=0 ; k<md->dimz; k++){
				if( navg[i][j][k]>0 ){
					md->xavg[i][j][k] = md->xavg[i][j][k]/(1.0*navg[i][j][k]);
					md->yavg[i][j][k] = md->yavg[i][j][k]/(1.0*navg[i][j][k]);
					md->zavg[i][j][k] = md->zavg[i][j][k]/(1.0*navg[i][j][k]);
				}
				else{
					md->xavg[i][j][k] = 0.5+i;
					md->yavg[i][j][k] = 0.5+j;
					md->zavg[i][j][k] = 0.5+k;
				}
	}	}	}
	
	// filter defects
	for( i=0; i<md->dimx; i++){ for( j=0; j<md->dimy; j++){ for( k=0; k<md->dimz; k++){
		idx1 = 0; idx2 = 0; 
		for( u=i-1; u<=i+1; u++ ){ for( v=j-1; v<=j+1; v++){ for( w=k-1; w<=k+1; w++){
			ui = u; vi = v; wi = w;
			if( ui<0 ){ ui = ui+md->dimx; } else if( ui>=md->dimx ){ ui = ui-md->dimx; }
			if( vi<0 ){ vi = vi+md->dimy; } else if( vi>=md->dimy ){ vi = vi-md->dimy; }
			if( wi<0 ){ wi = wi+md->dimz; } else if( wi>=md->dimz ){ wi = wi-md->dimz; }
			if( md->mapdefect[ui][vi][wi]>=0 ){ idx1 = idx1+1; }
		} } }
		for( u=i-3; u<=i+3; u++ ){ for( v=j-3; v<=j+3; v++){ for( w=k-3; w<=k+3; w++){
			ui = u; vi = v; wi = w;
			if( ui<0 ){ ui = ui+md->dimx; } else if( ui>=md->dimx ){ ui = ui-md->dimx; }
			if( vi<0 ){ vi = vi+md->dimy; } else if( vi>=md->dimy ){ vi = vi-md->dimy; }
			if( wi<0 ){ wi = wi+md->dimz; } else if( wi>=md->dimz ){ wi = wi-md->dimz; }
			if( md->mapdefect[ui][vi][wi]>=0 ){ idx2 = idx2+1; }
		} } }
		if( idx1==idx2 && idx1<=defect_thd_n ){
			for( u=i-1; u<=i+1; u++ ){ for( v=j-1; v<=j+1; v++){ for( w=k-1; w<=k+1; w++){
				ui = u; vi = v; wi = w;
				if( ui<0 ){ ui = ui+md->dimx; } else if( ui>=md->dimx ){ ui = ui-md->dimx; }
				if( vi<0 ){ vi = vi+md->dimy; } else if( vi>=md->dimy ){ vi = vi-md->dimy; }
				if( wi<0 ){ wi = wi+md->dimz; } else if( wi>=md->dimz ){ wi = wi-md->dimz; }
				md->mapdefect[ui][vi][wi]=-1;
			} } }
		}
	} } }
	
	// free array memory
	for (i=0; i<md->dimx; i++){
		for (j=0; j<md->dimy; j++){ delete [] navg[i][j]; }
		delete [] navg[i];
	}
	delete [] navg;
}



// calculate force
void force_cal(struct matlabData *imgs, struct mdrun *md){
	long i, listindex, atmindex, atmcount, atm1, atm2;
	int j, in, ip, jn, jp, kn, kp;
	int ic, jc, kc, ii, jj, kk;
	double xl, yl, zl;
	double dbi1, dbi2, dbo1, dbo2;
	double shfx, shfy, shfz;
	double cmpx, cmpy, cmpz, r1, r2, db1, db2, db3, frce;
	double dx, dy, dz;
	double rs;
	double v[2][2][2], vc[64];
	
	
	// damping force
	#pragma omp parallel for private(i) num_threads(md->ncpu)
	for(i=0;i < md->natm;i++){ 
		md->fxnew[i]=-dampcoeff*md->vx[i]; 
		md->fynew[i]=-dampcoeff*md->vy[i];
		md->fznew[i]=-dampcoeff*md->vz[i];
	}
	#pragma omp barrier

	// calculate force by structure potentials
	#pragma omp parallel for private(i,j,in,ip,jn,jp,kn,kp,xl,yl,zl,v,vc,dbi1,dbi2,dbo1,dbo2) num_threads(md->ncpu)
	for( i=0; i<md->natm; i++){
		in = int(floor(md->xnew[i])); jn = int(floor(md->ynew[i])); kn = int(floor(md->znew[i]));
		xl = md->xnew[i]-in; yl = md->ynew[i]-jn; zl = md->znew[i]-kn;
		if(itp_method==0){   // for trilinear interpolation
			ip = in+1; if(ip>=md->dimx) {ip=ip%(md->dimx);}
			jp = jn+1; if(jp>=md->dimy) {jp=jp%(md->dimy);}
			kp = kn+1; if(kp>=md->dimz) {kp=kp%(md->dimz);}
			v[0][0][0] = imgs->img3dg[in][jn][kn]; v[1][0][0] = imgs->img3dg[ip][jn][kn];
			v[0][1][0] = imgs->img3dg[in][jp][kn]; v[1][1][0] = imgs->img3dg[ip][jp][kn];
			v[0][0][1] = imgs->img3dg[in][jn][kp]; v[1][0][1] = imgs->img3dg[ip][jn][kp];
			v[0][1][1] = imgs->img3dg[in][jp][kp]; v[1][1][1] = imgs->img3dg[ip][jp][kp];
		}
		else if(itp_method==1) { for(j=0 ; j<64; j++){ vc[j] = md->intp3[in][jn][kn][j]; } }  // for tricubic interpolation
		// force x
	    dbi1 =  xl-delta; if(dbi1<0.0) {dbi1=0.0;}  dbi2 =  xl+delta; if(dbi2>1.0) {dbi2=1.0;}
		dbo1 = interp(dbi1, yl, zl, v, vc, itp_method); dbo2 = interp(dbi2, yl, zl, v, vc, itp_method);
		md->fxnew[i] = md->fxnew[i]+2.0*(dbo2-dbo1)/(dbi2-dbi1);				// potential V=-img3dg, so force is -(-dv/dr)
		// force y
	    dbi1 =  yl-delta; if(dbi1<0.0) {dbi1=0.0;}  dbi2 =  yl+delta; if(dbi2>1.0) {dbi2=1.0;}
		dbo1 = interp(xl, dbi1, zl, v, vc, itp_method); dbo2 = interp(xl, dbi2, zl, v, vc, itp_method);
		md->fynew[i] = md->fynew[i]+2.0*(dbo2-dbo1)/(dbi2-dbi1);
		// force z
	    dbi1 =  zl-delta; if(dbi1<0.0) {dbi1=0.0;}  dbi2 =  zl+delta; if(dbi2>1.0) {dbi2=1.0;}
		dbo1 = interp(xl, yl, dbi1, v, vc, itp_method); dbo2 = interp(xl, yl, dbi2, v, vc, itp_method);
		md->fznew[i] = md->fznew[i]+2.0*(dbo2-dbo1)/(dbi2-dbi1);
	}
	#pragma omp barrier	
	
	//calculate force by local spring
	#pragma omp parallel for private(i,dx,dy,dz) num_threads(md->ncpu)
	for( i=0; i<md->natm; i++){
		dx = md->xnew[i]-md->x0[i]; dy = md->ynew[i]-md->y0[i]; dz = md->znew[i]-md->z0[i];
		if(dx>0.5*md->dimx)        { dx = dx-md->dimx; }
		else if(dx<=-0.5*md->dimx) { dx = dx+md->dimx; }
		if(dy>0.5*md->dimy)        { dy = dy-md->dimy; }
		else if(dy<=-0.5*md->dimy) { dy = dy+md->dimy; }
		if(dz>0.5*md->dimz)        { dz = dz-md->dimz; }
		else if(dz<=-0.5*md->dimz) { dz = dz+md->dimz; }
		// force x
		md->fxnew[i] = md->fxnew[i]-1.0*md->k*dx;			
		// force y
		md->fynew[i] = md->fynew[i]-1.0*md->k*dy;
		// force z
		md->fznew[i] = md->fznew[i]-1.0*md->k*dz;
	}
	#pragma omp barrier		
	
	// calculate force by L-J potential
	#pragma omp parallel for private(i,ic,jc,kc,ii,jj,kk,in,jn,kn,shfx,shfy,shfz,listindex,atmindex,atmcount,cmpx,cmpy,cmpz,r1,r2,db1,db2,frce) num_threads(md->ncpu)
	for(i=0; i<md->natm; i++){
		ic = int(floor(md->xnew[i])); jc = int(floor(md->ynew[i])); kc = int(floor(md->znew[i]));
		for(ii=ic-neiggrid; ii<=(ic+neiggrid); ii++){ for(jj=jc-neiggrid; jj<=(jc+neiggrid); jj++){ for(kk=kc-neiggrid ;kk<=(kc+neiggrid); kk++) {
			in = ii; jn = jj; kn = kk; shfx=0.0; shfy=0.0; shfz=0.0;
			if(in<0)                { in=in+md->dimx; shfx=-md->dimx; }
			else if(in >= md->dimx) { in=in-md->dimx; shfx= md->dimx; }
			if(jn<0)                { jn=jn+md->dimy; shfy=-md->dimy; }
			else if(jn >= md->dimy) { jn=jn-md->dimy; shfy= md->dimy; }
			if(kn<0)                { kn=kn+md->dimz; shfz=-md->dimz; }
			else if(kn >= md->dimz) { kn=kn-md->dimz; shfz= md->dimz; }
			listindex = md->mapgrid[in][jn][kn];
			if(listindex>=0){
				for(atmcount=1; atmcount <= md->numlist[listindex]; atmcount++){
					atmindex = md->ilist[listindex][atmcount-1];
					if( i != atmindex ){
						cmpx=md->xnew[atmindex]+shfx-md->xnew[i]; 
						cmpy=md->ynew[atmindex]+shfy-md->ynew[i];
						cmpz=md->znew[atmindex]+shfz-md->znew[i];
						r2=cmpx*cmpx+cmpy*cmpy+cmpz*cmpz; 
						r1=sqrt(r2); if(r1==0.0){ r1=delta; printf("r1 is zero at atoms %d and %d\n",i,atmindex); }
						cmpx=cmpx/r1; cmpy=cmpy/r1; cmpz=cmpz/r1;
						// db1=pow(md->sig/r1,12.0); db2=pow(md->sig/r1,6.0);
						db1=md->sig/r1; db1=db1*db1; db1=db1*db1; db1=db1*db1*db1;
						db2=md->sig/r1; db2=db2*db2; db2=db2*db2*db2;
						frce=4.0*md->eps*( 12.0*db1-6.0*db2 )/r1;
						md->fxnew[i]=md->fxnew[i]-frce*cmpx;
						md->fynew[i]=md->fynew[i]-frce*cmpy;
						md->fznew[i]=md->fznew[i]-frce*cmpz;	
						if( r1>0.5*md->dimx || r1>0.5*md->dimy || r1>0.5*md->dimz){ printf("L-J calculation error!!\n"); }
					}
				}
			}
		}}}		
	}
	#pragma omp barrier
	
	// calculate bonding force
	#pragma omp parallel for private(i,j,atm1,atm2,db1,db2,db3,r1,rs,frce) num_threads(md->ncpu)
	for(i=0; i<md->natm; i++){
		for(j=md->bondij_bgn_end[i][0]; j<=md->bondij_bgn_end[i][1]; j++){
			atm1 = md->bondij[j][0]; atm2 = md->bondij[j][1];
			if( atm1 != atm2 ){
				db1 = md->xnew[atm2]-md->xnew[atm1]; 
				db2 = md->ynew[atm2]-md->ynew[atm1];
				db3 = md->znew[atm2]-md->znew[atm1];
				if(db1>0.5*md->dimx)        { db1 = db1-md->dimx; }
				else if(db1<=-0.5*md->dimx) { db1 = db1+md->dimx; }
				if(db2>0.5*md->dimy)        { db2 = db2-md->dimy; }
				else if(db2<=-0.5*md->dimy) { db2 = db2+md->dimy; }
				if(db3>0.5*md->dimz)        { db3 = db3-md->dimz; }
				else if(db3<=-0.5*md->dimz) { db3 = db3+md->dimz; }
				r1 = sqrt(db1*db1+db2*db2+db3*db3);
				if( r1==0.0 ){ r1=delta; printf("r1 is zero at atoms %d and %d\n",atm1,atm2); }
				if( r1>=(1.0-delta)*lmax) { rs=(1.0-delta); } else{ rs = r1/lmax; }
				rs = rs*rs; rs = rs*rs*rs;
				frce = 2.0*md->eps*rs/(1.0-rs)/r1;
				md->fxnew[i]=md->fxnew[i]+frce*db1;
				md->fynew[i]=md->fynew[i]+frce*db2;
				md->fznew[i]=md->fznew[i]+frce*db3;
				if( r1>0.5*md->dimx || r1>0.5*md->dimy || r1>0.5*md->dimz){ printf("Bonding calculation error!!\n"); }
			}
		}
	}
	#pragma omp barrier
}



// export atom xyz ( without extra adding atoms in subroutine "map_defect_add_atom" )
void export_xyz(struct mdrun *md, char fi){
	int i, j, k;
	long listindex, atmcount, atmindex;
	char fpname[99];
	FILE *fp;
	
	// set file name
	sprintf(fpname, "xyz_%c.txt",fi);
	fp = fopen(fpname, "w");
	for(i=0 ; i<md->dimx; i++){
		for(j=0 ; j<md->dimy; j++){
			for(k=0 ; k<md->dimz; k++){ 
				listindex = md->mapgrid[i][j][k];
				if(listindex>=0){
					for(atmcount=1; atmcount <= md->numlist[listindex]; atmcount++){
						atmindex = md->ilist[listindex][atmcount-1];
					    fprintf(fp,"%7.3f\t%7.3f\t%7.3f\n",md->x[atmindex],md->y[atmindex],md->z[atmindex]);
					}
				} 
	}	}	}
	fclose(fp);
}



// molecular dynamics
void md_cal(struct matlabData *imgs, struct mdrun *md){
	int i, j, k, ti, ibuf;
	long ii;
	double fdev, vdev, ff, vv;
	FILE *fp;
	
	
	srand((unsigned int)time(NULL));
	// initialize md arrays;
	ibuf = 0;
	for(i=0 ; i<md->dimx; i++){
		for(j=0 ; j<md->dimy; j++){
			for(k=0 ; k<md->dimz; k++){
				if(imgs->img3d[i][j][k]>0.5){ 
					md->x[ibuf]=i+0.25; md->y[ibuf]=j+0.25; md->z[ibuf]=k+0.25;	// img3dg(periodic potential) define values of grid corners, img3d(particle) define position in grid center. 
					md->xnew[ibuf]=i+0.25; md->ynew[ibuf]=j+0.25; md->znew[ibuf]=k+0.25;
					md->x0[ibuf]=i+0.25; md->y0[ibuf]=j+0.25; md->z0[ibuf]=k+0.25;
					md->fx[ibuf]=0.0;  md->fy[ibuf]=0.0;  md->fz[ibuf]=0.0;
					md->fxnew[ibuf]=0.0;  md->fynew[ibuf]=0.0;  md->fznew[ibuf]=0.0;
					md->vx[ibuf]= (((double) rand() / (RAND_MAX))-0.5)*2.0*sqrt(md->T);
					md->vy[ibuf]= (((double) rand() / (RAND_MAX))-0.5)*2.0*sqrt(md->T);
					md->vz[ibuf]= (((double) rand() / (RAND_MAX))-0.5)*2.0*sqrt(md->T);
					ibuf=ibuf+1;
					md->x[ibuf]=i+0.75; md->y[ibuf]=j+0.75; md->z[ibuf]=k+0.75;	// img3dg(periodic potential) define values of grid corners, img3d(particle) define position in grid center. 
					md->xnew[ibuf]=i+0.75; md->ynew[ibuf]=j+0.75; md->znew[ibuf]=k+0.75;
					md->x0[ibuf]=i+0.75; md->y0[ibuf]=j+0.75; md->z0[ibuf]=k+0.75;
					md->fx[ibuf]=0.0;  md->fy[ibuf]=0.0;  md->fz[ibuf]=0.0;
					md->fxnew[ibuf]=0.0;  md->fynew[ibuf]=0.0;  md->fznew[ibuf]=0.0;
					md->vx[ibuf]= (((double) rand() / (RAND_MAX))-0.5)*2.0*sqrt(md->T);
					md->vy[ibuf]= (((double) rand() / (RAND_MAX))-0.5)*2.0*sqrt(md->T);
					md->vz[ibuf]= (((double) rand() / (RAND_MAX))-0.5)*2.0*sqrt(md->T);
					ibuf=ibuf+1; 
				}
	}	}	}
	// vel_rescale(md, tauT);														// rescale velocity to meet assigne temperature md->T
	
	// filter defects and export initial atom xyz
	gen_neighbor_list(md);   // build md->mapgrid matrix
	map_defect(md);
	export_xyz(md, 'i');
	
	// build bonding list, with given initial md->mapgrid matrix 
	gen_bond_list(md);
	
	// time evolution
	lmax_ini = sqrt(3.0)*(0.5+neiggrid);
	lmax_stable = lmax_ini;
	for(ti=0; ti<nstep; ti++){
		
		// define lmax
        lmax = lmax_ini+(lmax_stable-lmax_ini)*ti/(nstep-1.0);
		
		// update potision
		// vel_rescale(md, dt);
		#pragma omp parallel for private(ii) num_threads(md->ncpu)
		for (ii=0; ii<md->natm; ii++){
			md->xnew[ii] = md->x[ii]+dt*md->vx[ii]+0.5*dt*dt*md->fx[ii];
			md->ynew[ii] = md->y[ii]+dt*md->vy[ii]+0.5*dt*dt*md->fy[ii];
			md->znew[ii] = md->z[ii]+dt*md->vz[ii]+0.5*dt*dt*md->fz[ii];
		}
		#pragma omp barrier
		
		// periodic boundary
		#pragma omp parallel for private(ii) num_threads(md->ncpu)
		for (ii=0; ii<md->natm; ii++){
			if(md->xnew[ii] < 0.0)           { md->xnew[ii]=md->xnew[ii]+md->dimx; }
			else if(md->xnew[ii] >= md->dimx){ md->xnew[ii]=md->xnew[ii]-md->dimx; }
			if(md->ynew[ii] < 0.0)           { md->ynew[ii]=md->ynew[ii]+md->dimy; }
			else if(md->ynew[ii] >= md->dimy){ md->ynew[ii]=md->ynew[ii]-md->dimy; }
			if(md->znew[ii] < 0.0)           { md->znew[ii]=md->znew[ii]+md->dimz; }
			else if(md->znew[ii] >= md->dimz){ md->znew[ii]=md->znew[ii]-md->dimz; }
		}
		#pragma omp barrier
		
		// calculate new force
		gen_neighbor_list(md);
		force_cal(imgs, md);
		
		// update velocity
		#pragma omp parallel for private(ii) num_threads(md->ncpu)
		for (ii=0; ii<md->natm; ii++){
			md->vxnew[ii] = md->vx[ii]+0.5*dt*( md->fx[ii]+md->fxnew[ii] );
			md->vynew[ii] = md->vy[ii]+0.5*dt*( md->fy[ii]+md->fynew[ii] );
			md->vznew[ii] = md->vz[ii]+0.5*dt*( md->fz[ii]+md->fznew[ii] );
		}
		#pragma omp barrier
		
		// show message in log file
		if( (ti%10)==0 ){ 
			fdev = 0.0; vdev = 0.0;
			for (ii=0; ii<md->natm; ii++){
				ff = sqrt( pow(md->fx[ii]-md->fxnew[ii],2.0)+pow(md->fy[ii]-md->fynew[ii],2.0)+pow(md->fz[ii]-md->fznew[ii],2.0) );
				vv = sqrt( pow(md->vx[ii]-md->vxnew[ii],2.0)+pow(md->vy[ii]-md->vynew[ii],2.0)+pow(md->vz[ii]-md->vznew[ii],2.0) );
				if (ff>fdev) { fdev = ff;}
				if (vv>vdev) { vdev = vv;}
			}
			if(ti==0) { fp=fopen("log.txt","w");}
			else {fp=fopen("log.txt","a");}
			fprintf(fp,"MD running at step:\t%9d,\tdf=%12.5f,\tdv=%12.5f\n",ti,fdev,vdev);
			fclose(fp); 
		}
		
		// iteration copy
		#pragma omp parallel for private(ii) num_threads(md->ncpu)
		for (ii=0; ii<md->natm; ii++){
			md->x[ii]=md->xnew[ii]; md->y[ii]=md->ynew[ii]; md->z[ii]=md->znew[ii];
			md->vx[ii]=md->vxnew[ii]; md->vy[ii]=md->vynew[ii]; md->vz[ii]=md->vznew[ii];
			md->fx[ii]=md->fxnew[ii]; md->fy[ii]=md->fynew[ii]; md->fz[ii]=md->fznew[ii];
		}
		#pragma omp barrier
	}

	// filter defects and export final atom xyz
	gen_neighbor_list(md); // build md->mapgrid matrix
	// map_defect(md);
	map_defect_add_atom(md);
	export_xyz(md, 'o');
}

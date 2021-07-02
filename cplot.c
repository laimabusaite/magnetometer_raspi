#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <VG/openvg.h>
#include <VG/vgu.h>
#include <fontinfo.h>
#include <shapes.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv)
{
	int width, height;
	char s[3];
    int fontsize = 30;
    int angle = 60; // x = x + 0.5*z*cosa, y = y + 0.5*z*sina
	int x0, y0, e;
	int x1, y1;
	float Bx, By, Bz;
	int scale;
	char result[100] = "B = (";
	char rez1[20];
	int i;

	//FILE *f;
	//f = fopen("B_values.dat", "r");
	//fscanf(f, "%f", &Bx);
	//fscanf(f, "%f", &By);
	//fscanf(f, "%f", &Bz);
	//printf("Bxyz = (%.2f, %.2f, %.2f)\n",Bx, By, Bz);
	//fclose(f);

	Bx = atof(argv[1]);
	By = atof(argv[2]);
	Bz = atof(argv[3]);

	init(&width, &height);
	Start(width, height);
    //printf("%d %d",width, height);
	x0=width/2;
	y0=height/2;
	e=1000;
	scale=width/(2*10);
	Background(0,0,0);
	Stroke(0,255,0,1);
	StrokeWidth(10);
	Line(x0,y0,x0+e,y0);
	Line(x0,y0,x0,y0+e);
	Line(x0,y0,x0+0.5*e*0.5,y0+0.5*e*0.866);
	StrokeWidth(2);
	for (i=0;i<10;i++)
	{
		Line(x0+100*i,y0,x0+100*i,y0+20);
	}
	for (i=0;i<10;i++)
	{
		Line(x0,y0+100*i,x0+20,y0+100*i);
	}
	for (i=0;i<10;i++)
	{
		Line(x0+0.5*100*i*0.5,y0+0.5*100*i*0.866,x0+0.5*100*i*0.5,y0+0.5*100*i*0.866+20);
	}
	Fill(255,255,255,1);
	TextMid(width-fontsize,height/2-2*fontsize,"x",SerifTypeface,fontsize);
	TextMid(width/2-fontsize,height-fontsize,"y",SerifTypeface,fontsize);
	TextMid(x0+0.5*e*0.5-fontsize,y0+0.5*e*0.866-0*fontsize,"z",SerifTypeface,fontsize);
	TextMid(x0-fontsize,y0,"0",SerifTypeface,fontsize);
	TextMid(x0+5*scale+0.5*fontsize,y0-2*fontsize,"5",SerifTypeface,fontsize);
	TextMid(x0-2*fontsize,y0+5*scale,"5",SerifTypeface,fontsize);
	TextMid(x0+0.5*5*scale*0.5-0.5*fontsize,y0+0.5*5*scale*0.866+fontsize,"5",SerifTypeface,fontsize);
	//Bx=10;
	//By=10;
	//Bz=10;
	Stroke(255,0,0,1);
	StrokeWidth(20);
	Line(x0,y0,x0+scale*(Bx+0.5*Bz*0.5),y0+scale*(By+0.5*Bz*0.866));
	Fill(255,255,255,1);
	//printf(result);
	sprintf(rez1,"%.2f",Bx);
	strcat(result,rez1);
	strcat(result,", ");
	sprintf(rez1,"%.2f",By);
	strcat(result,rez1);
	strcat(result,", ");
	sprintf(rez1,"%.2f",Bz);
	strcat(result,rez1);
	strcat(result,") G        |B| = ");
	sprintf(rez1,"%.2f",sqrt(Bx*Bx+By*By+Bz*Bz));
	strcat(result,rez1);
	strcat(result," G");
	//printf(result);
	TextMid(500,100,result,SerifTypeface,fontsize);
	End();
	//fgets(s,2,stdin);
	usleep(800000);
	finish();
	exit(0);
}

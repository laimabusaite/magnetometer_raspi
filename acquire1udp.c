#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "rp.h"
#include <signal.h>

#include <stdint.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <errno.h>
#include <string.h>
#include <linux/spi/spidev.h>
#include <linux/types.h>

//#include <stdio.h>
//#include <stdlib.h>
//#include <unistd.h>
//#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>

#define PORT 8080
#define MAXLINE 1024


int running = 1;

void signalhandler(int signum)
{
	running = 0;
}

int main(int argc, char **argv){

	signal(SIGINT, signalhandler);

    //char *data = "HELLO WOLRD!";
	char data[16];

int sockfd;
char buffer[MAXLINE];
//char *hello = "hello";
struct sockaddr_in servaddr, cliaddr;

if ((sockfd=socket(AF_INET,SOCK_DGRAM,0))<0)
{
perror("socket initialization failed");
exit(EXIT_FAILURE);
}

memset(&servaddr,0,sizeof(servaddr));
memset(&cliaddr,0,sizeof(cliaddr));

servaddr.sin_family = AF_INET;
servaddr.sin_addr.s_addr = inet_addr("192.168.0.101");
servaddr.sin_port = htons(PORT);

if (bind(sockfd, (const struct sockaddr *)&servaddr, sizeof(servaddr))<0)
{
perror("bind fail");
exit(EXIT_FAILURE);
}

int len, n;
len = sizeof(cliaddr);

//int n;
n=recvfrom(sockfd, (char *)buffer, MAXLINE, MSG_WAITALL, (struct sockaddr *) &cliaddr, &len);
//n=recvfrom(sockfd, (char *)buffer, MAXLINE, MSG_WAITALL, (struct sockaddr *) &cliaddr, sizeof(cliaddr));
buffer[n] = '\0';
printf("client: %s\n", buffer);
//for (int i=0;i<500000;i++)
//{
//printf("asd1");
//sprintf(buffer,"%d",i);
//printf("%s",buffer);
//}
//printf("sent\n");


        /* Print error, if rp_Init() function failed */
        if(rp_Init() != RP_OK){
               fprintf(stderr, "Rp api init failed!\n");
        }

        //rp_GenReset();
        //rp_GenWaveform(RP_CH_1, RP_WAVEFORM_DC);
        //rp_GenOutEnable(RP_CH_1);

	//rp_GenAmp(RP_CH_1, 0.0);

        uint32_t buff_size = 16;
        float *buff1 = (float *)malloc(buff_size * sizeof(float));
	float *buff2 = (float *)malloc(buff_size * sizeof(float));

        //rp_AcqReset();
        rp_AcqSetDecimation(RP_DEC_1);
        rp_AcqSetTriggerDelay(0);

	rp_AcqSetGain(RP_CH_1, RP_HIGH);
	rp_AcqSetGain(RP_CH_2, RP_HIGH);

	int led=0;
	led+=RP_LED0;

	while(running)
	{
	rp_DpinSetState(led, RP_HIGH);

        rp_AcqReset();
	rp_AcqStart();

        /* After acquisition is started some time delay is needed in order to acquire fresh samples in to buffer*/
        /* Here we have used time delay of one second but you can calculate exact value taking in to account buffer*/
        /*length and smaling rate*/

        usleep(300);
        rp_AcqSetTriggerSrc(RP_TRIG_SRC_NOW);
        rp_acq_trig_state_t state = RP_TRIG_STATE_TRIGGERED;

        while(1){
                rp_AcqGetTriggerState(&state);
                if(state == RP_TRIG_STATE_TRIGGERED){
                usleep(300);
                break;
                }
        }

        rp_AcqGetOldestDataV(RP_CH_1, &buff_size, buff1);
	rp_AcqGetOldestDataV(RP_CH_2, &buff_size, buff2);

        int i;
	float division=0;
        for(i = 0; i < buff_size; i++){
                //printf("%f %f %f\n", buff1[i], buff2[i],buff1[i]/buff2[i]);
		//rp_GenAmp(RP_CH_1, buff1[i]/buff2[i]);
		division += buff1[i]/buff2[i];
        }
	//printf("%f\n",division/16);
	//division = (20*division/16)-19;
	division = division/16;
	//rp_GenAmp(RP_CH_1, division);
	gcvt(division, 7, data);
sendto(sockfd, (const char *)data, strlen(data), MSG_CONFIRM, (const struct sockaddr *) &cliaddr, sizeof(cliaddr));
	rp_DpinSetState(led, RP_LOW);
	led+=1;
	if(led>7)
	{
		led=0;
		led+=RP_LED0;
	}

	}

        /* Releasing resources */
	printf("exit\n");
	free(buff1);
	free(buff2);
    rp_Release();

      return 0;
}

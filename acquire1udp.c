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

#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>

#define PORT 8080
#define MAXLINE 1024

int running = 1;

void signalhandler(int signum) // Handle the Ctrl+C signal to terminate the program
{
	running = 0;
}

int main(int argc, char **argv){

	signal(SIGINT, signalhandler);

	char data[16];

    int sockfd; // Set up the UDP socket
    char buffer[MAXLINE];

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
        perror("Bind fail error");
        exit(EXIT_FAILURE);
    }

    int len, n;
    len = sizeof(cliaddr);

    n=recvfrom(sockfd, (char *)buffer, MAXLINE, MSG_WAITALL, (struct sockaddr *) &cliaddr, &len);
    buffer[n] = '\0';
    printf("client: %s\n", buffer);

    /* Print error, if rp_Init() function failed */
    if(rp_Init() != RP_OK)
    {
        fprintf(stderr, "Rp api init failed!\n");
    }

    uint32_t buff_size = 16;
    float *buff1 = (float *)malloc(buff_size * sizeof(float));
	float *buff2 = (float *)malloc(buff_size * sizeof(float));

    // Set up the ADC data acquisition

    rp_AcqSetDecimation(RP_DEC_1);
    rp_AcqSetTriggerDelay(0);

	rp_AcqSetGain(RP_CH_1, RP_HIGH);
	rp_AcqSetGain(RP_CH_2, RP_HIGH);

	int led=0; // Set up measurement status LEDs
	led+=RP_LED0;

	while(running)
	{
    	rp_DpinSetState(led, RP_HIGH);

        rp_AcqReset();
	    rp_AcqStart();

        /* After acquisition is started some time delay is needed in order to acquire fresh samples in to buffer*/

        usleep(300);
        rp_AcqSetTriggerSrc(RP_TRIG_SRC_NOW);
        rp_acq_trig_state_t state = RP_TRIG_STATE_TRIGGERED;

        while(1)
        {
            rp_AcqGetTriggerState(&state);
            if(state == RP_TRIG_STATE_TRIGGERED)
            {
                usleep(300);
                break;
            }
        }

        rp_AcqGetOldestDataV(RP_CH_1, &buff_size, buff1); // Measure data on channel 1 (fluorescence intensity)
	    rp_AcqGetOldestDataV(RP_CH_2, &buff_size, buff2); // Measure data on channel 2 (laser power)

        int i;
	    float division=0;
        for(i = 0; i < buff_size; i++)
        {
            division += buff1[i]/buff2[i]; // Divide the fluorescence signal with the laser power to mitigate noise from laser power fluctuations
        }
	    division = division/16;
	    gcvt(division, 7, data);
        sendto(sockfd, (const char *)data, strlen(data), MSG_CONFIRM, (const struct sockaddr *) &cliaddr, sizeof(cliaddr)); // Send data to Raspberry Pi using UDP
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

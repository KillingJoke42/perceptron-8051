#include<reg51.h>
#include "weights.h"

void main(void)
{
	float w = weight();
	float b = bias();
	while(1)
	{
		P2 = 0;
		P2 = (unsigned char)(w*0x01 + b);
	}
}
#include <stdio.h>
#include <iostream>

using namespace std;

/////////////////////////////////////////
//	delete blank between words
/////////////////////////////////////////
int main(int argc, char* argv[])
{
	if(argc != 3)
	{
		cerr << "usage: input output" << endl;
		return 0;
	}
	
	FILE* in = fopen(argv[1], "r");
	FILE* out = fopen(argv[2], "w");
	
	if(in == NULL || out == NULL)
	{
		cerr << "error: can't open file!" << endl;
		return 0;
	}	
	
	int ret = 0;
	int state = 0;
	while((ret=fgetc(in)) != -1)
	{
		char ch = (char)ret;
		switch(state)
		{
		case 0:
			if(ch >= 0 && ch != '\n' && ch != ' ')
			{
				state = 1;	
			}
			break;
		case 1:
			if(ch < 0 || ch == '\n')
			{
				state = 0;
			}
			else if(ch == ' ')
			{
				state = 2;
			}
			break;
		case 2:
			if(ch >= 0 && ch != '\n' && ch != ' ')
			{
				fputc(' ', out);
				state = 1;
			}
			else if(ch < 0 || ch == '\n')
			{
				state = 0;
			}
			break;
		default:
			break;
		}
		if(ch != ' ')
		{
			fputc(ch, out);
		}
	}
	fclose(in);	
	fclose(out);
	return 0;
}


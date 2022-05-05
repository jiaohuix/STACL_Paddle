#include <stdio.h>
#include <iostream>

using namespace std;

///////////////////////////////////////
//	add blank between utf8 characters
///////////////////////////////////////
int main(int argc, char* argv[])
{
    //if(argc != 3)
	//{
	//	cerr << "usage: input output" << endl;
	//	return 0;
	//}
	
	//FILE* in = fopen(argv[1], "r");
	//FILE* out = fopen(argv[2], "w");
	
	//if(in == NULL || out == NULL)
	//{
	//	cerr << "error: can't open file!" << endl;
	//	return 0;
	//}	
	
	int ch = 0;
	bool flag = true;
	while( (ch=fgetc(stdin)) != -1)
	{
		if((char)ch >= 0)
		{
			if(flag)
			{
				fputc(' ', stdout);
				flag = false;
			}
			fputc(ch, stdout);
		}
		else
		{
			flag = true;
			char c = (char)ch;		
			fputc(' ', stdout);
			fputc(ch, stdout);
			while(((c=(c<<1)) & 0x80) != 0)
 			{
				ch = fgetc(stdin);
				if(ch == -1)
				{
					cerr << "error!" << endl;
					return 0;
				}
				fputc(ch, stdout);				
			}
		}
	}
	//fclose(in);	
	//fclose(out);
	return 0;
}

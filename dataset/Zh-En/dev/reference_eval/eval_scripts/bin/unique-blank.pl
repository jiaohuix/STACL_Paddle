#!/usr/bin/perl

while(<STDIN>)
{
	chomp;
	s/ +/ /g;
	s/^ //g;
	s/ $//g;
	print "$_\n";
}


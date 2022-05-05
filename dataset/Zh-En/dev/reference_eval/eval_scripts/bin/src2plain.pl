#!/usr/bin/perl

while(<STDIN>)
{
    chomp;
    if(m/^<seg id=\d+>.*<\/seg>$/)
    {
        s/^<seg id=\d+>(.*)<\/seg>$/$1/g;
        print "$_\n";
    }
}


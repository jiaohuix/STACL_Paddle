#!/usr/bin/perl

if(scalar(@ARGV) < 1)
{
    die "usage: config < input > output\n";
}

my %params = ();
foreach $s (split(/ /, @ARGV[0]))
{
    if($s =~ m/\=/)
    {
        my ($k, $v) = split(/\=/, $s);
        $param{$k} = $v;
    }
}

while($s=<STDIN>)
{
    chomp $s;
    if(exists($param{$s}))
    {
        print "$s\n";
        print "$param{$s}\n";
        delete($param{$s});
        <STDIN>;
    }
    else
    {
        print "$s\n";
    }
}

foreach $k (keys(%param))
{
    print "$k\n";
    print "$param{$k}\n";
}


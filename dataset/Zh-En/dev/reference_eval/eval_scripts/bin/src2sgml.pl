#!/usr/bin/perl

use Getopt::Std;
use Getopt::Long;

my $opt_i;
my $opt_d;
my $opt_s;
GetOptions('id=s', \$opt_i, 'doc=s', \$opt_d, 'src=s', \$opt_s);

my $id = defined($opt_i) ? $opt_i : "set";
my $doc = defined($opt_d) ? $opt_d : "document";
my $src = defined($opt_s) ? $opt_s : "source";

print '<srcset setid="'.$id.'" srclang="'.$src.'">'."\n";
print '<DOC docid="'.$doc.'">'."\n";

my $id = 1;

while( <> )
{
	chomp;
	print '<seg id=';
	printf( "%03d>", $id++ );
	print;
	print '</seg>'."\n";
}

print '</DOC>'."\n";
print '</srcset>'."\n";

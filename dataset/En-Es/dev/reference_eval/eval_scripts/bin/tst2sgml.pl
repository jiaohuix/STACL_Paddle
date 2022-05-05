#!/usr/bin/perl

use Getopt::Std;
use Getopt::Long;

my $opt_i;
my $opt_d;
my $opt_s;
my $opt_t;
GetOptions('id=s', \$opt_i, 'doc=s', \$opt_d, 'src=s', \$opt_s, 'trg=s', \$opt_t, 'sys=s', \$opt_e, );

my $id = defined($opt_i) ? $opt_i : "set";
my $doc = defined($opt_d) ? $opt_d : "document";
my $src = defined($opt_s) ? $opt_s : "source";
my $trg = defined($opt_t) ? $opt_t : "target";
my $sys = defined($opt_e) ? $opt_e : "system";

print '<tstset setid="'.$id.'" srclang="'.$src.'" trglang="'.$trg.'">'."\n";
print '<DOC docid="'.$doc.'" sysid="'.$sys.'">'."\n";

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
print '</tstset>'."\n";

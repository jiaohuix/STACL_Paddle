#!/usr/bin/perl

my $state = 0;
my $sysid = "";
while($s = <STDIN>)
{
    chomp $s;
    if($state == 0)
    {
        if($s =~ m/^<refset.*$/)
        {
            print "$s\n";
            $state = 1;
        }
        else
        {
            die "Error: unknown format in line $s\n";
        }
    }
    elsif($state == 1)
    {
        if($s =~ m/^.*docid=.*sysid=\"\w+\".*$/)
        {
            $sysid = $s;
            $sysid =~ s/^.*sysid=\"(\w+)\".*$/$1/g;
            print "$s\n";
            $state = 2;
        }
        else
        {
            die "Error: unknown format in line $s\n";
        }
    }
    elsif($state == 2)
    {
        if($s =~ m/^<seg id=\d+>.*<\/seg>$/ or $s =~ m/^<seg id=\"\d+\">.*<\/seg>$/)
        {
            print "$s\n";
        }
        elsif($s =~ m/^<\/DOC>/i)
        {
            print "$s\n";
            $state = 3;
        }
        elsif($s ne "\<poster\>" and $s ne "\<\/poster\>")
        {
            die "Error: unknown format in line $s\n";
        }
    }
    elsif($state == 3)
    {
        if($s =~ m/^.*docid=.*sysid=\"\w+\".*$/)
        {
            my $id = $s;
            $id =~ s/^.*sysid=\"(\w+)\".*$/$1/g;
            if($id eq $sysid)
            {
                print "$s\n";
                $state = 2;
            }
        }
        elsif($s =~ m/^<\/refset>$/)
        {
            print "$s\n";
            last;
        }
    }
}


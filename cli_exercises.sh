#!/bin/sh
echo "grep '\<A.*\>' /usr/share/dict/words | wc -l"
grep '\<A.*\>' /usr/share/dict/words | wc -l
echo "grep '\<A.*\>' /usr/share/dict/words | wc -l"
grep '\<[Aa].*\>' /usr/share/dict/words | wc -l
echo "grep '\<.*ing\>' /usr/share/dict/words | wc -l"
grep '\<.*ing\>' /usr/share/dict/words | wc -l
echo "grep '\<[Aa].*ing\>' /usr/share/dict/words | wc -l"
grep '\<[Aa].*ing\>' /usr/share/dict/words | wc -l
echo "grep '^.\{4\}$' /usr/share/dict/words | wc -l"
grep '^.\{4\}$' /usr/share/dict/words | wc -l
echo "grep '^.*[aeiou].*[aeiou].*[aeiou].*[aeiou].*' /usr/share/dict/words | wc -l"
grep '^.*[aeiou].*[aeiou].*[aeiou].*[aeiou].*' /usr/share/dict/words | wc -l
echo "grep '\([aeiou]\)\1' /usr/share/dict/words | wc -l"
grep '\([aeiou]\)\1' /usr/share/dict/words | wc -l
echo "grep '^.*q[^u].*$\|^.*q$' /usr/share/dict/words"
grep '^.*q[^u].*$\|^.*q$' /usr/share/dict/words
echo "cut -c 1 /usr/share/dict/words | sort | uniq -c"
cut -c 1 /usr/share/dict/words | sort | uniq -c
echo "cut -c 1 /usr/share/dict/words | tr a-z A-Z | sort | uniq -c"
cut -c 1 /usr/share/dict/words | tr a-z A-Z | sort | uniq -c

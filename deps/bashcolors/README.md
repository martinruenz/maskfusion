Bash colors
===========

The library defines verbal constants for ANSII color codes.
Now you can use words to tell bash what color you want to use for output.

Also, it defines functions to quickly output colored text.
As well as a function to display nice table of all colors.

Updated to allow for iterative output of colors by sequencing the color functions


Examples
--------

Print "foobar" with red foreground:
```sh
$ clr_red "foobar"
```

Print "foobar" with green background:
```sh
$ clr_greenb "foobar"
```

Print "foobar" with cyan foreground and bold font:
```sh
$ clr_bold clr_cyan "foobar"
```

Stack functions on a single output to produce complex color configurations:
```sh
$ clr_red clr_blueb clr_reverse clr_underscore "foobar"
```

Using the -n switch, the output can remain on a single line, allowing for changing colors across a line of text:
```sh
$ clr_redb clr_black "RED" -n; clr_blueb clr_black "BLUE" -n; clr_greenb clr_black "GREEN";
```

or Print "foobar" manually, using variables:
```sh
$ clr_escape "foobar" $CLR_BOLD $CLR_CYAN
```

or any other code:
```sh
$ clr_escape "foobar" 1 36
```


Quick start
------------

1. Download source
```sh
$ curl https://raw.githubusercontent.com/mercuriev/bash_colors/master/bash_colors.sh > .bash_colors
```

2. Load into your script
```sh
$ source .bash_colors
```


Variables
---------

**$CLR_***

Variables for each supported color code. E.g. $CLR_WHITE, $CLR_BLACK
Call **clr_dump** for full list.

**$CLR_ESC**
is a special variable for escape code (\033) followed by [ character.


Functions
---------

**clr_dump**
Output table of available colors, functions and variables.

**clr_*** _string_
Functions for each supported color. E.g. clr_white, clr_black
See **clr_dump** for list.

**fn_exists**
General bash function to check if string is a valid function

**clr_layer**
Recursive function allowing for muliple color functions in a sequence

**clr_escape** _string_ _$CLR_ ...
escape _string_ with given colors. Latest colors overwrites previous.

**clr_reset**
Reset formatting. Useful for custom usage of $CLR_* variable series.


See also
--------

console_codes(4)
[Advanced Bash Scripting Guide](http://tldp.org/LDP/abs/html/colorizing.html)

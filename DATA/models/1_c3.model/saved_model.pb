ШЂ:
О╗
D
AddV2
x"T
y"T
z"T"
Ttype:
2	ђљ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ѕ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
Џ
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(ѕ

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
@
Softplus
features"T
activations"T"
Ttype:
2
┴
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ѕе
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.9.12unknown8Х▄,
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
v
value_head/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namevalue_head/bias
o
#value_head/bias/Read/ReadVariableOpReadVariableOpvalue_head/bias*
_output_shapes
:*
dtype0

value_head/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	т,*"
shared_namevalue_head/kernel
x
%value_head/kernel/Read/ReadVariableOpReadVariableOpvalue_head/kernel*
_output_shapes
:	т,*
dtype0
v
border_off/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameborder_off/bias
o
#border_off/bias/Read/ReadVariableOpReadVariableOpborder_off/bias*
_output_shapes
:*
dtype0
є
border_off/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameborder_off/kernel

%border_off/kernel/Read/ReadVariableOpReadVariableOpborder_off/kernel*&
_output_shapes
:*
dtype0
ё
policy_aggregator/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namepolicy_aggregator/bias
}
*policy_aggregator/bias/Read/ReadVariableOpReadVariableOppolicy_aggregator/bias*
_output_shapes
:*
dtype0
ћ
policy_aggregator/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namepolicy_aggregator/kernel
Ї
,policy_aggregator/kernel/Read/ReadVariableOpReadVariableOppolicy_aggregator/kernel*&
_output_shapes
:*
dtype0
ђ
contract_20_3x3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namecontract_20_3x3/bias
y
(contract_20_3x3/bias/Read/ReadVariableOpReadVariableOpcontract_20_3x3/bias*
_output_shapes
:*
dtype0
љ
contract_20_3x3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namecontract_20_3x3/kernel
Ѕ
*contract_20_3x3/kernel/Read/ReadVariableOpReadVariableOpcontract_20_3x3/kernel*&
_output_shapes
: *
dtype0
|
expand_20_5x5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameexpand_20_5x5/bias
u
&expand_20_5x5/bias/Read/ReadVariableOpReadVariableOpexpand_20_5x5/bias*
_output_shapes
: *
dtype0
ї
expand_20_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *%
shared_nameexpand_20_5x5/kernel
Ё
(expand_20_5x5/kernel/Read/ReadVariableOpReadVariableOpexpand_20_5x5/kernel*&
_output_shapes
:	 *
dtype0
ђ
contract_19_3x3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namecontract_19_3x3/bias
y
(contract_19_3x3/bias/Read/ReadVariableOpReadVariableOpcontract_19_3x3/bias*
_output_shapes
:*
dtype0
љ
contract_19_3x3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namecontract_19_3x3/kernel
Ѕ
*contract_19_3x3/kernel/Read/ReadVariableOpReadVariableOpcontract_19_3x3/kernel*&
_output_shapes
: *
dtype0
|
expand_19_5x5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameexpand_19_5x5/bias
u
&expand_19_5x5/bias/Read/ReadVariableOpReadVariableOpexpand_19_5x5/bias*
_output_shapes
: *
dtype0
ї
expand_19_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *%
shared_nameexpand_19_5x5/kernel
Ё
(expand_19_5x5/kernel/Read/ReadVariableOpReadVariableOpexpand_19_5x5/kernel*&
_output_shapes
:	 *
dtype0
ђ
contract_18_3x3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namecontract_18_3x3/bias
y
(contract_18_3x3/bias/Read/ReadVariableOpReadVariableOpcontract_18_3x3/bias*
_output_shapes
:*
dtype0
љ
contract_18_3x3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namecontract_18_3x3/kernel
Ѕ
*contract_18_3x3/kernel/Read/ReadVariableOpReadVariableOpcontract_18_3x3/kernel*&
_output_shapes
: *
dtype0
|
expand_18_5x5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameexpand_18_5x5/bias
u
&expand_18_5x5/bias/Read/ReadVariableOpReadVariableOpexpand_18_5x5/bias*
_output_shapes
: *
dtype0
ї
expand_18_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *%
shared_nameexpand_18_5x5/kernel
Ё
(expand_18_5x5/kernel/Read/ReadVariableOpReadVariableOpexpand_18_5x5/kernel*&
_output_shapes
:	 *
dtype0
ђ
contract_17_3x3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namecontract_17_3x3/bias
y
(contract_17_3x3/bias/Read/ReadVariableOpReadVariableOpcontract_17_3x3/bias*
_output_shapes
:*
dtype0
љ
contract_17_3x3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namecontract_17_3x3/kernel
Ѕ
*contract_17_3x3/kernel/Read/ReadVariableOpReadVariableOpcontract_17_3x3/kernel*&
_output_shapes
: *
dtype0
|
expand_17_5x5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameexpand_17_5x5/bias
u
&expand_17_5x5/bias/Read/ReadVariableOpReadVariableOpexpand_17_5x5/bias*
_output_shapes
: *
dtype0
ї
expand_17_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *%
shared_nameexpand_17_5x5/kernel
Ё
(expand_17_5x5/kernel/Read/ReadVariableOpReadVariableOpexpand_17_5x5/kernel*&
_output_shapes
:	 *
dtype0
ђ
contract_16_3x3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namecontract_16_3x3/bias
y
(contract_16_3x3/bias/Read/ReadVariableOpReadVariableOpcontract_16_3x3/bias*
_output_shapes
:*
dtype0
љ
contract_16_3x3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namecontract_16_3x3/kernel
Ѕ
*contract_16_3x3/kernel/Read/ReadVariableOpReadVariableOpcontract_16_3x3/kernel*&
_output_shapes
: *
dtype0
|
expand_16_5x5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameexpand_16_5x5/bias
u
&expand_16_5x5/bias/Read/ReadVariableOpReadVariableOpexpand_16_5x5/bias*
_output_shapes
: *
dtype0
ї
expand_16_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *%
shared_nameexpand_16_5x5/kernel
Ё
(expand_16_5x5/kernel/Read/ReadVariableOpReadVariableOpexpand_16_5x5/kernel*&
_output_shapes
:	 *
dtype0
ђ
contract_15_3x3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namecontract_15_3x3/bias
y
(contract_15_3x3/bias/Read/ReadVariableOpReadVariableOpcontract_15_3x3/bias*
_output_shapes
:*
dtype0
љ
contract_15_3x3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namecontract_15_3x3/kernel
Ѕ
*contract_15_3x3/kernel/Read/ReadVariableOpReadVariableOpcontract_15_3x3/kernel*&
_output_shapes
: *
dtype0
|
expand_15_5x5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameexpand_15_5x5/bias
u
&expand_15_5x5/bias/Read/ReadVariableOpReadVariableOpexpand_15_5x5/bias*
_output_shapes
: *
dtype0
ї
expand_15_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *%
shared_nameexpand_15_5x5/kernel
Ё
(expand_15_5x5/kernel/Read/ReadVariableOpReadVariableOpexpand_15_5x5/kernel*&
_output_shapes
:	 *
dtype0
ђ
contract_14_3x3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namecontract_14_3x3/bias
y
(contract_14_3x3/bias/Read/ReadVariableOpReadVariableOpcontract_14_3x3/bias*
_output_shapes
:*
dtype0
љ
contract_14_3x3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namecontract_14_3x3/kernel
Ѕ
*contract_14_3x3/kernel/Read/ReadVariableOpReadVariableOpcontract_14_3x3/kernel*&
_output_shapes
: *
dtype0
|
expand_14_5x5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameexpand_14_5x5/bias
u
&expand_14_5x5/bias/Read/ReadVariableOpReadVariableOpexpand_14_5x5/bias*
_output_shapes
: *
dtype0
ї
expand_14_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *%
shared_nameexpand_14_5x5/kernel
Ё
(expand_14_5x5/kernel/Read/ReadVariableOpReadVariableOpexpand_14_5x5/kernel*&
_output_shapes
:	 *
dtype0
ђ
contract_13_3x3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namecontract_13_3x3/bias
y
(contract_13_3x3/bias/Read/ReadVariableOpReadVariableOpcontract_13_3x3/bias*
_output_shapes
:*
dtype0
љ
contract_13_3x3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namecontract_13_3x3/kernel
Ѕ
*contract_13_3x3/kernel/Read/ReadVariableOpReadVariableOpcontract_13_3x3/kernel*&
_output_shapes
: *
dtype0
|
expand_13_5x5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameexpand_13_5x5/bias
u
&expand_13_5x5/bias/Read/ReadVariableOpReadVariableOpexpand_13_5x5/bias*
_output_shapes
: *
dtype0
ї
expand_13_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *%
shared_nameexpand_13_5x5/kernel
Ё
(expand_13_5x5/kernel/Read/ReadVariableOpReadVariableOpexpand_13_5x5/kernel*&
_output_shapes
:	 *
dtype0
ђ
contract_12_3x3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namecontract_12_3x3/bias
y
(contract_12_3x3/bias/Read/ReadVariableOpReadVariableOpcontract_12_3x3/bias*
_output_shapes
:*
dtype0
љ
contract_12_3x3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namecontract_12_3x3/kernel
Ѕ
*contract_12_3x3/kernel/Read/ReadVariableOpReadVariableOpcontract_12_3x3/kernel*&
_output_shapes
: *
dtype0
|
expand_12_5x5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameexpand_12_5x5/bias
u
&expand_12_5x5/bias/Read/ReadVariableOpReadVariableOpexpand_12_5x5/bias*
_output_shapes
: *
dtype0
ї
expand_12_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *%
shared_nameexpand_12_5x5/kernel
Ё
(expand_12_5x5/kernel/Read/ReadVariableOpReadVariableOpexpand_12_5x5/kernel*&
_output_shapes
:	 *
dtype0
ђ
contract_11_3x3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namecontract_11_3x3/bias
y
(contract_11_3x3/bias/Read/ReadVariableOpReadVariableOpcontract_11_3x3/bias*
_output_shapes
:*
dtype0
љ
contract_11_3x3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namecontract_11_3x3/kernel
Ѕ
*contract_11_3x3/kernel/Read/ReadVariableOpReadVariableOpcontract_11_3x3/kernel*&
_output_shapes
: *
dtype0
|
expand_11_5x5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameexpand_11_5x5/bias
u
&expand_11_5x5/bias/Read/ReadVariableOpReadVariableOpexpand_11_5x5/bias*
_output_shapes
: *
dtype0
ї
expand_11_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *%
shared_nameexpand_11_5x5/kernel
Ё
(expand_11_5x5/kernel/Read/ReadVariableOpReadVariableOpexpand_11_5x5/kernel*&
_output_shapes
:	 *
dtype0
ђ
contract_10_3x3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namecontract_10_3x3/bias
y
(contract_10_3x3/bias/Read/ReadVariableOpReadVariableOpcontract_10_3x3/bias*
_output_shapes
:*
dtype0
љ
contract_10_3x3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namecontract_10_3x3/kernel
Ѕ
*contract_10_3x3/kernel/Read/ReadVariableOpReadVariableOpcontract_10_3x3/kernel*&
_output_shapes
: *
dtype0
|
expand_10_5x5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameexpand_10_5x5/bias
u
&expand_10_5x5/bias/Read/ReadVariableOpReadVariableOpexpand_10_5x5/bias*
_output_shapes
: *
dtype0
ї
expand_10_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *%
shared_nameexpand_10_5x5/kernel
Ё
(expand_10_5x5/kernel/Read/ReadVariableOpReadVariableOpexpand_10_5x5/kernel*&
_output_shapes
:	 *
dtype0
~
contract_9_3x3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namecontract_9_3x3/bias
w
'contract_9_3x3/bias/Read/ReadVariableOpReadVariableOpcontract_9_3x3/bias*
_output_shapes
:*
dtype0
ј
contract_9_3x3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_namecontract_9_3x3/kernel
Є
)contract_9_3x3/kernel/Read/ReadVariableOpReadVariableOpcontract_9_3x3/kernel*&
_output_shapes
: *
dtype0
z
expand_9_5x5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameexpand_9_5x5/bias
s
%expand_9_5x5/bias/Read/ReadVariableOpReadVariableOpexpand_9_5x5/bias*
_output_shapes
: *
dtype0
і
expand_9_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *$
shared_nameexpand_9_5x5/kernel
Ѓ
'expand_9_5x5/kernel/Read/ReadVariableOpReadVariableOpexpand_9_5x5/kernel*&
_output_shapes
:	 *
dtype0
~
contract_8_3x3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namecontract_8_3x3/bias
w
'contract_8_3x3/bias/Read/ReadVariableOpReadVariableOpcontract_8_3x3/bias*
_output_shapes
:*
dtype0
ј
contract_8_3x3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_namecontract_8_3x3/kernel
Є
)contract_8_3x3/kernel/Read/ReadVariableOpReadVariableOpcontract_8_3x3/kernel*&
_output_shapes
: *
dtype0
z
expand_8_5x5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameexpand_8_5x5/bias
s
%expand_8_5x5/bias/Read/ReadVariableOpReadVariableOpexpand_8_5x5/bias*
_output_shapes
: *
dtype0
і
expand_8_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *$
shared_nameexpand_8_5x5/kernel
Ѓ
'expand_8_5x5/kernel/Read/ReadVariableOpReadVariableOpexpand_8_5x5/kernel*&
_output_shapes
:	 *
dtype0
~
contract_7_3x3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namecontract_7_3x3/bias
w
'contract_7_3x3/bias/Read/ReadVariableOpReadVariableOpcontract_7_3x3/bias*
_output_shapes
:*
dtype0
ј
contract_7_3x3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_namecontract_7_3x3/kernel
Є
)contract_7_3x3/kernel/Read/ReadVariableOpReadVariableOpcontract_7_3x3/kernel*&
_output_shapes
: *
dtype0
z
expand_7_5x5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameexpand_7_5x5/bias
s
%expand_7_5x5/bias/Read/ReadVariableOpReadVariableOpexpand_7_5x5/bias*
_output_shapes
: *
dtype0
і
expand_7_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *$
shared_nameexpand_7_5x5/kernel
Ѓ
'expand_7_5x5/kernel/Read/ReadVariableOpReadVariableOpexpand_7_5x5/kernel*&
_output_shapes
:	 *
dtype0
~
contract_6_3x3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namecontract_6_3x3/bias
w
'contract_6_3x3/bias/Read/ReadVariableOpReadVariableOpcontract_6_3x3/bias*
_output_shapes
:*
dtype0
ј
contract_6_3x3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_namecontract_6_3x3/kernel
Є
)contract_6_3x3/kernel/Read/ReadVariableOpReadVariableOpcontract_6_3x3/kernel*&
_output_shapes
: *
dtype0
z
expand_6_5x5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameexpand_6_5x5/bias
s
%expand_6_5x5/bias/Read/ReadVariableOpReadVariableOpexpand_6_5x5/bias*
_output_shapes
: *
dtype0
і
expand_6_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *$
shared_nameexpand_6_5x5/kernel
Ѓ
'expand_6_5x5/kernel/Read/ReadVariableOpReadVariableOpexpand_6_5x5/kernel*&
_output_shapes
:	 *
dtype0
~
contract_5_3x3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namecontract_5_3x3/bias
w
'contract_5_3x3/bias/Read/ReadVariableOpReadVariableOpcontract_5_3x3/bias*
_output_shapes
:*
dtype0
ј
contract_5_3x3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_namecontract_5_3x3/kernel
Є
)contract_5_3x3/kernel/Read/ReadVariableOpReadVariableOpcontract_5_3x3/kernel*&
_output_shapes
: *
dtype0
z
expand_5_5x5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameexpand_5_5x5/bias
s
%expand_5_5x5/bias/Read/ReadVariableOpReadVariableOpexpand_5_5x5/bias*
_output_shapes
: *
dtype0
і
expand_5_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *$
shared_nameexpand_5_5x5/kernel
Ѓ
'expand_5_5x5/kernel/Read/ReadVariableOpReadVariableOpexpand_5_5x5/kernel*&
_output_shapes
:	 *
dtype0
~
contract_4_3x3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namecontract_4_3x3/bias
w
'contract_4_3x3/bias/Read/ReadVariableOpReadVariableOpcontract_4_3x3/bias*
_output_shapes
:*
dtype0
ј
contract_4_3x3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_namecontract_4_3x3/kernel
Є
)contract_4_3x3/kernel/Read/ReadVariableOpReadVariableOpcontract_4_3x3/kernel*&
_output_shapes
: *
dtype0
z
expand_4_5x5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameexpand_4_5x5/bias
s
%expand_4_5x5/bias/Read/ReadVariableOpReadVariableOpexpand_4_5x5/bias*
_output_shapes
: *
dtype0
і
expand_4_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *$
shared_nameexpand_4_5x5/kernel
Ѓ
'expand_4_5x5/kernel/Read/ReadVariableOpReadVariableOpexpand_4_5x5/kernel*&
_output_shapes
:	 *
dtype0
~
contract_3_3x3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namecontract_3_3x3/bias
w
'contract_3_3x3/bias/Read/ReadVariableOpReadVariableOpcontract_3_3x3/bias*
_output_shapes
:*
dtype0
ј
contract_3_3x3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_namecontract_3_3x3/kernel
Є
)contract_3_3x3/kernel/Read/ReadVariableOpReadVariableOpcontract_3_3x3/kernel*&
_output_shapes
: *
dtype0
z
expand_3_5x5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameexpand_3_5x5/bias
s
%expand_3_5x5/bias/Read/ReadVariableOpReadVariableOpexpand_3_5x5/bias*
_output_shapes
: *
dtype0
і
expand_3_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *$
shared_nameexpand_3_5x5/kernel
Ѓ
'expand_3_5x5/kernel/Read/ReadVariableOpReadVariableOpexpand_3_5x5/kernel*&
_output_shapes
:	 *
dtype0
~
contract_2_3x3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namecontract_2_3x3/bias
w
'contract_2_3x3/bias/Read/ReadVariableOpReadVariableOpcontract_2_3x3/bias*
_output_shapes
:*
dtype0
ј
contract_2_3x3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_namecontract_2_3x3/kernel
Є
)contract_2_3x3/kernel/Read/ReadVariableOpReadVariableOpcontract_2_3x3/kernel*&
_output_shapes
: *
dtype0
z
expand_2_5x5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameexpand_2_5x5/bias
s
%expand_2_5x5/bias/Read/ReadVariableOpReadVariableOpexpand_2_5x5/bias*
_output_shapes
: *
dtype0
і
expand_2_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *$
shared_nameexpand_2_5x5/kernel
Ѓ
'expand_2_5x5/kernel/Read/ReadVariableOpReadVariableOpexpand_2_5x5/kernel*&
_output_shapes
:	 *
dtype0
~
contract_1_5x5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namecontract_1_5x5/bias
w
'contract_1_5x5/bias/Read/ReadVariableOpReadVariableOpcontract_1_5x5/bias*
_output_shapes
:*
dtype0
Ј
contract_1_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_namecontract_1_5x5/kernel
ѕ
)contract_1_5x5/kernel/Read/ReadVariableOpReadVariableOpcontract_1_5x5/kernel*'
_output_shapes
:ђ*
dtype0
є
heuristic_priority/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameheuristic_priority/bias

+heuristic_priority/bias/Read/ReadVariableOpReadVariableOpheuristic_priority/bias*
_output_shapes
:*
dtype0
Ќ
heuristic_priority/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:│**
shared_nameheuristic_priority/kernel
љ
-heuristic_priority/kernel/Read/ReadVariableOpReadVariableOpheuristic_priority/kernel*'
_output_shapes
:│*
dtype0

expand_1_11x11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*$
shared_nameexpand_1_11x11/bias
x
'expand_1_11x11/bias/Read/ReadVariableOpReadVariableOpexpand_1_11x11/bias*
_output_shapes	
:ђ*
dtype0
Ј
expand_1_11x11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameexpand_1_11x11/kernel
ѕ
)expand_1_11x11/kernel/Read/ReadVariableOpReadVariableOpexpand_1_11x11/kernel*'
_output_shapes
:ђ*
dtype0
Є
heuristic_detector/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:│*(
shared_nameheuristic_detector/bias
ђ
+heuristic_detector/bias/Read/ReadVariableOpReadVariableOpheuristic_detector/bias*
_output_shapes	
:│*
dtype0
Ќ
heuristic_detector/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:│**
shared_nameheuristic_detector/kernel
љ
-heuristic_detector/kernel/Read/ReadVariableOpReadVariableOpheuristic_detector/kernel*'
_output_shapes
:│*
dtype0

NoOpNoOp
Ош
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Љш
valueєшBѓш BЩЗ
▀
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
layer-12
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
layer-16
layer-17
layer_with_weights-10
layer-18
layer_with_weights-11
layer-19
layer-20
layer-21
layer_with_weights-12
layer-22
layer_with_weights-13
layer-23
layer-24
layer-25
layer_with_weights-14
layer-26
layer_with_weights-15
layer-27
layer-28
layer-29
layer_with_weights-16
layer-30
 layer_with_weights-17
 layer-31
!layer-32
"layer-33
#layer_with_weights-18
#layer-34
$layer_with_weights-19
$layer-35
%layer-36
&layer-37
'layer_with_weights-20
'layer-38
(layer_with_weights-21
(layer-39
)layer-40
*layer-41
+layer_with_weights-22
+layer-42
,layer_with_weights-23
,layer-43
-layer-44
.layer-45
/layer_with_weights-24
/layer-46
0layer_with_weights-25
0layer-47
1layer-48
2layer-49
3layer_with_weights-26
3layer-50
4layer_with_weights-27
4layer-51
5layer-52
6layer-53
7layer_with_weights-28
7layer-54
8layer_with_weights-29
8layer-55
9layer-56
:layer-57
;layer_with_weights-30
;layer-58
<layer_with_weights-31
<layer-59
=layer-60
>layer-61
?layer_with_weights-32
?layer-62
@layer_with_weights-33
@layer-63
Alayer-64
Blayer-65
Clayer_with_weights-34
Clayer-66
Dlayer_with_weights-35
Dlayer-67
Elayer-68
Flayer-69
Glayer_with_weights-36
Glayer-70
Hlayer_with_weights-37
Hlayer-71
Ilayer-72
Jlayer-73
Klayer_with_weights-38
Klayer-74
Llayer_with_weights-39
Llayer-75
Mlayer-76
Nlayer-77
Olayer_with_weights-40
Olayer-78
Player_with_weights-41
Player-79
Qlayer-80
Rlayer_with_weights-42
Rlayer-81
Slayer-82
Tlayer_with_weights-43
Tlayer-83
Ulayer-84
Vlayer-85
Wlayer-86
Xlayer-87
Ylayer_with_weights-44
Ylayer-88
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses
`_default_save_signature
a	optimizer
bloss
c
signatures*
* 
╚
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses

jkernel
kbias
 l_jit_compiled_convolution_op*
╚
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses

skernel
tbias
 u_jit_compiled_convolution_op*
╚
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses

|kernel
}bias
 ~_jit_compiled_convolution_op*
л
	variables
ђtrainable_variables
Ђregularization_losses
ѓ	keras_api
Ѓ__call__
+ё&call_and_return_all_conditional_losses
Ёkernel
	єbias
!Є_jit_compiled_convolution_op*
ћ
ѕ	variables
Ѕtrainable_variables
іregularization_losses
І	keras_api
ї__call__
+Ї&call_and_return_all_conditional_losses* 
Л
ј	variables
Јtrainable_variables
љregularization_losses
Љ	keras_api
њ__call__
+Њ&call_and_return_all_conditional_losses
ћkernel
	Ћbias
!ќ_jit_compiled_convolution_op*
Л
Ќ	variables
ўtrainable_variables
Ўregularization_losses
џ	keras_api
Џ__call__
+ю&call_and_return_all_conditional_losses
Юkernel
	ъbias
!Ъ_jit_compiled_convolution_op*
ћ
а	variables
Аtrainable_variables
бregularization_losses
Б	keras_api
ц__call__
+Ц&call_and_return_all_conditional_losses* 
ћ
д	variables
Дtrainable_variables
еregularization_losses
Е	keras_api
ф__call__
+Ф&call_and_return_all_conditional_losses* 
Л
г	variables
Гtrainable_variables
«regularization_losses
»	keras_api
░__call__
+▒&call_and_return_all_conditional_losses
▓kernel
	│bias
!┤_jit_compiled_convolution_op*
Л
х	variables
Хtrainable_variables
иregularization_losses
И	keras_api
╣__call__
+║&call_and_return_all_conditional_losses
╗kernel
	╝bias
!й_jit_compiled_convolution_op*
ћ
Й	variables
┐trainable_variables
└regularization_losses
┴	keras_api
┬__call__
+├&call_and_return_all_conditional_losses* 
ћ
─	variables
┼trainable_variables
кregularization_losses
К	keras_api
╚__call__
+╔&call_and_return_all_conditional_losses* 
Л
╩	variables
╦trainable_variables
╠regularization_losses
═	keras_api
╬__call__
+¤&call_and_return_all_conditional_losses
лkernel
	Лbias
!м_jit_compiled_convolution_op*
Л
М	variables
нtrainable_variables
Нregularization_losses
о	keras_api
О__call__
+п&call_and_return_all_conditional_losses
┘kernel
	┌bias
!█_jit_compiled_convolution_op*
ћ
▄	variables
Пtrainable_variables
яregularization_losses
▀	keras_api
Я__call__
+р&call_and_return_all_conditional_losses* 
ћ
Р	variables
сtrainable_variables
Сregularization_losses
т	keras_api
Т__call__
+у&call_and_return_all_conditional_losses* 
Л
У	variables
жtrainable_variables
Жregularization_losses
в	keras_api
В__call__
+ь&call_and_return_all_conditional_losses
Ьkernel
	№bias
!­_jit_compiled_convolution_op*
Л
ы	variables
Ыtrainable_variables
зregularization_losses
З	keras_api
ш__call__
+Ш&call_and_return_all_conditional_losses
эkernel
	Эbias
!щ_jit_compiled_convolution_op*
ћ
Щ	variables
чtrainable_variables
Чregularization_losses
§	keras_api
■__call__
+ &call_and_return_all_conditional_losses* 
ћ
ђ	variables
Ђtrainable_variables
ѓregularization_losses
Ѓ	keras_api
ё__call__
+Ё&call_and_return_all_conditional_losses* 
Л
є	variables
Єtrainable_variables
ѕregularization_losses
Ѕ	keras_api
і__call__
+І&call_and_return_all_conditional_losses
їkernel
	Їbias
!ј_jit_compiled_convolution_op*
Л
Ј	variables
љtrainable_variables
Љregularization_losses
њ	keras_api
Њ__call__
+ћ&call_and_return_all_conditional_losses
Ћkernel
	ќbias
!Ќ_jit_compiled_convolution_op*
ћ
ў	variables
Ўtrainable_variables
џregularization_losses
Џ	keras_api
ю__call__
+Ю&call_and_return_all_conditional_losses* 
ћ
ъ	variables
Ъtrainable_variables
аregularization_losses
А	keras_api
б__call__
+Б&call_and_return_all_conditional_losses* 
Л
ц	variables
Цtrainable_variables
дregularization_losses
Д	keras_api
е__call__
+Е&call_and_return_all_conditional_losses
фkernel
	Фbias
!г_jit_compiled_convolution_op*
Л
Г	variables
«trainable_variables
»regularization_losses
░	keras_api
▒__call__
+▓&call_and_return_all_conditional_losses
│kernel
	┤bias
!х_jit_compiled_convolution_op*
ћ
Х	variables
иtrainable_variables
Иregularization_losses
╣	keras_api
║__call__
+╗&call_and_return_all_conditional_losses* 
ћ
╝	variables
йtrainable_variables
Йregularization_losses
┐	keras_api
└__call__
+┴&call_and_return_all_conditional_losses* 
Л
┬	variables
├trainable_variables
─regularization_losses
┼	keras_api
к__call__
+К&call_and_return_all_conditional_losses
╚kernel
	╔bias
!╩_jit_compiled_convolution_op*
Л
╦	variables
╠trainable_variables
═regularization_losses
╬	keras_api
¤__call__
+л&call_and_return_all_conditional_losses
Лkernel
	мbias
!М_jit_compiled_convolution_op*
ћ
н	variables
Нtrainable_variables
оregularization_losses
О	keras_api
п__call__
+┘&call_and_return_all_conditional_losses* 
ћ
┌	variables
█trainable_variables
▄regularization_losses
П	keras_api
я__call__
+▀&call_and_return_all_conditional_losses* 
Л
Я	variables
рtrainable_variables
Рregularization_losses
с	keras_api
С__call__
+т&call_and_return_all_conditional_losses
Тkernel
	уbias
!У_jit_compiled_convolution_op*
Л
ж	variables
Жtrainable_variables
вregularization_losses
В	keras_api
ь__call__
+Ь&call_and_return_all_conditional_losses
№kernel
	­bias
!ы_jit_compiled_convolution_op*
ћ
Ы	variables
зtrainable_variables
Зregularization_losses
ш	keras_api
Ш__call__
+э&call_and_return_all_conditional_losses* 
ћ
Э	variables
щtrainable_variables
Щregularization_losses
ч	keras_api
Ч__call__
+§&call_and_return_all_conditional_losses* 
Л
■	variables
 trainable_variables
ђregularization_losses
Ђ	keras_api
ѓ__call__
+Ѓ&call_and_return_all_conditional_losses
ёkernel
	Ёbias
!є_jit_compiled_convolution_op*
Л
Є	variables
ѕtrainable_variables
Ѕregularization_losses
і	keras_api
І__call__
+ї&call_and_return_all_conditional_losses
Їkernel
	јbias
!Ј_jit_compiled_convolution_op*
ћ
љ	variables
Љtrainable_variables
њregularization_losses
Њ	keras_api
ћ__call__
+Ћ&call_and_return_all_conditional_losses* 
ћ
ќ	variables
Ќtrainable_variables
ўregularization_losses
Ў	keras_api
џ__call__
+Џ&call_and_return_all_conditional_losses* 
Л
ю	variables
Юtrainable_variables
ъregularization_losses
Ъ	keras_api
а__call__
+А&call_and_return_all_conditional_losses
бkernel
	Бbias
!ц_jit_compiled_convolution_op*
Л
Ц	variables
дtrainable_variables
Дregularization_losses
е	keras_api
Е__call__
+ф&call_and_return_all_conditional_losses
Фkernel
	гbias
!Г_jit_compiled_convolution_op*
ћ
«	variables
»trainable_variables
░regularization_losses
▒	keras_api
▓__call__
+│&call_and_return_all_conditional_losses* 
ћ
┤	variables
хtrainable_variables
Хregularization_losses
и	keras_api
И__call__
+╣&call_and_return_all_conditional_losses* 
Л
║	variables
╗trainable_variables
╝regularization_losses
й	keras_api
Й__call__
+┐&call_and_return_all_conditional_losses
└kernel
	┴bias
!┬_jit_compiled_convolution_op*
Л
├	variables
─trainable_variables
┼regularization_losses
к	keras_api
К__call__
+╚&call_and_return_all_conditional_losses
╔kernel
	╩bias
!╦_jit_compiled_convolution_op*
ћ
╠	variables
═trainable_variables
╬regularization_losses
¤	keras_api
л__call__
+Л&call_and_return_all_conditional_losses* 
ћ
м	variables
Мtrainable_variables
нregularization_losses
Н	keras_api
о__call__
+О&call_and_return_all_conditional_losses* 
Л
п	variables
┘trainable_variables
┌regularization_losses
█	keras_api
▄__call__
+П&call_and_return_all_conditional_losses
яkernel
	▀bias
!Я_jit_compiled_convolution_op*
Л
р	variables
Рtrainable_variables
сregularization_losses
С	keras_api
т__call__
+Т&call_and_return_all_conditional_losses
уkernel
	Уbias
!ж_jit_compiled_convolution_op*
ћ
Ж	variables
вtrainable_variables
Вregularization_losses
ь	keras_api
Ь__call__
+№&call_and_return_all_conditional_losses* 
ћ
­	variables
ыtrainable_variables
Ыregularization_losses
з	keras_api
З__call__
+ш&call_and_return_all_conditional_losses* 
Л
Ш	variables
эtrainable_variables
Эregularization_losses
щ	keras_api
Щ__call__
+ч&call_and_return_all_conditional_losses
Чkernel
	§bias
!■_jit_compiled_convolution_op*
Л
 	variables
ђtrainable_variables
Ђregularization_losses
ѓ	keras_api
Ѓ__call__
+ё&call_and_return_all_conditional_losses
Ёkernel
	єbias
!Є_jit_compiled_convolution_op*
ћ
ѕ	variables
Ѕtrainable_variables
іregularization_losses
І	keras_api
ї__call__
+Ї&call_and_return_all_conditional_losses* 
ћ
ј	variables
Јtrainable_variables
љregularization_losses
Љ	keras_api
њ__call__
+Њ&call_and_return_all_conditional_losses* 
Л
ћ	variables
Ћtrainable_variables
ќregularization_losses
Ќ	keras_api
ў__call__
+Ў&call_and_return_all_conditional_losses
џkernel
	Џbias
!ю_jit_compiled_convolution_op*
Л
Ю	variables
ъtrainable_variables
Ъregularization_losses
а	keras_api
А__call__
+б&call_and_return_all_conditional_losses
Бkernel
	цbias
!Ц_jit_compiled_convolution_op*
ћ
д	variables
Дtrainable_variables
еregularization_losses
Е	keras_api
ф__call__
+Ф&call_and_return_all_conditional_losses* 
ћ
г	variables
Гtrainable_variables
«regularization_losses
»	keras_api
░__call__
+▒&call_and_return_all_conditional_losses* 
Л
▓	variables
│trainable_variables
┤regularization_losses
х	keras_api
Х__call__
+и&call_and_return_all_conditional_losses
Иkernel
	╣bias
!║_jit_compiled_convolution_op*
Л
╗	variables
╝trainable_variables
йregularization_losses
Й	keras_api
┐__call__
+└&call_and_return_all_conditional_losses
┴kernel
	┬bias
!├_jit_compiled_convolution_op*
ћ
─	variables
┼trainable_variables
кregularization_losses
К	keras_api
╚__call__
+╔&call_and_return_all_conditional_losses* 
ћ
╩	variables
╦trainable_variables
╠regularization_losses
═	keras_api
╬__call__
+¤&call_and_return_all_conditional_losses* 
Л
л	variables
Лtrainable_variables
мregularization_losses
М	keras_api
н__call__
+Н&call_and_return_all_conditional_losses
оkernel
	Оbias
!п_jit_compiled_convolution_op*
Л
┘	variables
┌trainable_variables
█regularization_losses
▄	keras_api
П__call__
+я&call_and_return_all_conditional_losses
▀kernel
	Яbias
!р_jit_compiled_convolution_op*
ћ
Р	variables
сtrainable_variables
Сregularization_losses
т	keras_api
Т__call__
+у&call_and_return_all_conditional_losses* 
ћ
У	variables
жtrainable_variables
Жregularization_losses
в	keras_api
В__call__
+ь&call_and_return_all_conditional_losses* 
Л
Ь	variables
№trainable_variables
­regularization_losses
ы	keras_api
Ы__call__
+з&call_and_return_all_conditional_losses
Зkernel
	шbias
!Ш_jit_compiled_convolution_op*
Л
э	variables
Эtrainable_variables
щregularization_losses
Щ	keras_api
ч__call__
+Ч&call_and_return_all_conditional_losses
§kernel
	■bias
! _jit_compiled_convolution_op*
ћ
ђ	variables
Ђtrainable_variables
ѓregularization_losses
Ѓ	keras_api
ё__call__
+Ё&call_and_return_all_conditional_losses* 
ћ
є	variables
Єtrainable_variables
ѕregularization_losses
Ѕ	keras_api
і__call__
+І&call_and_return_all_conditional_losses* 
Л
ї	variables
Їtrainable_variables
јregularization_losses
Ј	keras_api
љ__call__
+Љ&call_and_return_all_conditional_losses
њkernel
	Њbias
!ћ_jit_compiled_convolution_op*
Л
Ћ	variables
ќtrainable_variables
Ќregularization_losses
ў	keras_api
Ў__call__
+џ&call_and_return_all_conditional_losses
Џkernel
	юbias
!Ю_jit_compiled_convolution_op*
ћ
ъ	variables
Ъtrainable_variables
аregularization_losses
А	keras_api
б__call__
+Б&call_and_return_all_conditional_losses* 
ћ
ц	variables
Цtrainable_variables
дregularization_losses
Д	keras_api
е__call__
+Е&call_and_return_all_conditional_losses* 
Л
ф	variables
Фtrainable_variables
гregularization_losses
Г	keras_api
«__call__
+»&call_and_return_all_conditional_losses
░kernel
	▒bias
!▓_jit_compiled_convolution_op*
Л
│	variables
┤trainable_variables
хregularization_losses
Х	keras_api
и__call__
+И&call_and_return_all_conditional_losses
╣kernel
	║bias
!╗_jit_compiled_convolution_op*
ћ
╝	variables
йtrainable_variables
Йregularization_losses
┐	keras_api
└__call__
+┴&call_and_return_all_conditional_losses* 
Л
┬	variables
├trainable_variables
─regularization_losses
┼	keras_api
к__call__
+К&call_and_return_all_conditional_losses
╚kernel
	╔bias
!╩_jit_compiled_convolution_op*
ћ
╦	variables
╠trainable_variables
═regularization_losses
╬	keras_api
¤__call__
+л&call_and_return_all_conditional_losses* 
Л
Л	variables
мtrainable_variables
Мregularization_losses
н	keras_api
Н__call__
+о&call_and_return_all_conditional_losses
Оkernel
	пbias
!┘_jit_compiled_convolution_op*
ћ
┌	variables
█trainable_variables
▄regularization_losses
П	keras_api
я__call__
+▀&call_and_return_all_conditional_losses* 
ћ
Я	variables
рtrainable_variables
Рregularization_losses
с	keras_api
С__call__
+т&call_and_return_all_conditional_losses* 

Т	keras_api* 
ћ
у	variables
Уtrainable_variables
жregularization_losses
Ж	keras_api
в__call__
+В&call_and_return_all_conditional_losses* 
«
ь	variables
Ьtrainable_variables
№regularization_losses
­	keras_api
ы__call__
+Ы&call_and_return_all_conditional_losses
зkernel
	Зbias*
ъ
j0
k1
s2
t3
|4
}5
Ё6
є7
ћ8
Ћ9
Ю10
ъ11
▓12
│13
╗14
╝15
л16
Л17
┘18
┌19
Ь20
№21
э22
Э23
ї24
Ї25
Ћ26
ќ27
ф28
Ф29
│30
┤31
╚32
╔33
Л34
м35
Т36
у37
№38
­39
ё40
Ё41
Ї42
ј43
б44
Б45
Ф46
г47
└48
┴49
╔50
╩51
я52
▀53
у54
У55
Ч56
§57
Ё58
є59
џ60
Џ61
Б62
ц63
И64
╣65
┴66
┬67
о68
О69
▀70
Я71
З72
ш73
§74
■75
њ76
Њ77
Џ78
ю79
░80
▒81
╣82
║83
╚84
╔85
О86
п87
з88
З89*
В
s0
t1
Ё2
є3
ћ4
Ћ5
Ю6
ъ7
▓8
│9
╗10
╝11
л12
Л13
┘14
┌15
Ь16
№17
э18
Э19
ї20
Ї21
Ћ22
ќ23
ф24
Ф25
│26
┤27
╚28
╔29
Л30
м31
Т32
у33
№34
­35
ё36
Ё37
Ї38
ј39
б40
Б41
Ф42
г43
└44
┴45
╔46
╩47
я48
▀49
у50
У51
Ч52
§53
Ё54
є55
џ56
Џ57
Б58
ц59
И60
╣61
┴62
┬63
о64
О65
▀66
Я67
З68
ш69
§70
■71
њ72
Њ73
Џ74
ю75
░76
▒77
╣78
║79
╚80
╔81
з82
З83*
* 
х
шnon_trainable_variables
Шlayers
эmetrics
 Эlayer_regularization_losses
щlayer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
`_default_save_signature
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses*

Щtrace_0
чtrace_1* 

Чtrace_0
§trace_1* 
* 
* 
* 

■serving_default* 

j0
k1*
* 
* 
ў
 non_trainable_variables
ђlayers
Ђmetrics
 ѓlayer_regularization_losses
Ѓlayer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses*

ёtrace_0* 

Ёtrace_0* 
ic
VARIABLE_VALUEheuristic_detector/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEheuristic_detector/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

s0
t1*

s0
t1*
* 
ў
єnon_trainable_variables
Єlayers
ѕmetrics
 Ѕlayer_regularization_losses
іlayer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses*

Іtrace_0* 

їtrace_0* 
e_
VARIABLE_VALUEexpand_1_11x11/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEexpand_1_11x11/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

|0
}1*
* 
* 
ў
Їnon_trainable_variables
јlayers
Јmetrics
 љlayer_regularization_losses
Љlayer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses*

њtrace_0* 

Њtrace_0* 
ic
VARIABLE_VALUEheuristic_priority/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEheuristic_priority/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

Ё0
є1*

Ё0
є1*
* 
Ю
ћnon_trainable_variables
Ћlayers
ќmetrics
 Ќlayer_regularization_losses
ўlayer_metrics
	variables
ђtrainable_variables
Ђregularization_losses
Ѓ__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses*

Ўtrace_0* 

џtrace_0* 
e_
VARIABLE_VALUEcontract_1_5x5/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEcontract_1_5x5/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
ю
Џnon_trainable_variables
юlayers
Юmetrics
 ъlayer_regularization_losses
Ъlayer_metrics
ѕ	variables
Ѕtrainable_variables
іregularization_losses
ї__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses* 

аtrace_0* 

Аtrace_0* 

ћ0
Ћ1*

ћ0
Ћ1*
* 
ъ
бnon_trainable_variables
Бlayers
цmetrics
 Цlayer_regularization_losses
дlayer_metrics
ј	variables
Јtrainable_variables
љregularization_losses
њ__call__
+Њ&call_and_return_all_conditional_losses
'Њ"call_and_return_conditional_losses*

Дtrace_0* 

еtrace_0* 
c]
VARIABLE_VALUEexpand_2_5x5/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEexpand_2_5x5/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

Ю0
ъ1*

Ю0
ъ1*
* 
ъ
Еnon_trainable_variables
фlayers
Фmetrics
 гlayer_regularization_losses
Гlayer_metrics
Ќ	variables
ўtrainable_variables
Ўregularization_losses
Џ__call__
+ю&call_and_return_all_conditional_losses
'ю"call_and_return_conditional_losses*

«trace_0* 

»trace_0* 
e_
VARIABLE_VALUEcontract_2_3x3/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEcontract_2_3x3/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
ю
░non_trainable_variables
▒layers
▓metrics
 │layer_regularization_losses
┤layer_metrics
а	variables
Аtrainable_variables
бregularization_losses
ц__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses* 

хtrace_0* 

Хtrace_0* 
* 
* 
* 
ю
иnon_trainable_variables
Иlayers
╣metrics
 ║layer_regularization_losses
╗layer_metrics
д	variables
Дtrainable_variables
еregularization_losses
ф__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses* 

╝trace_0* 

йtrace_0* 

▓0
│1*

▓0
│1*
* 
ъ
Йnon_trainable_variables
┐layers
└metrics
 ┴layer_regularization_losses
┬layer_metrics
г	variables
Гtrainable_variables
«regularization_losses
░__call__
+▒&call_and_return_all_conditional_losses
'▒"call_and_return_conditional_losses*

├trace_0* 

─trace_0* 
c]
VARIABLE_VALUEexpand_3_5x5/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEexpand_3_5x5/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

╗0
╝1*

╗0
╝1*
* 
ъ
┼non_trainable_variables
кlayers
Кmetrics
 ╚layer_regularization_losses
╔layer_metrics
х	variables
Хtrainable_variables
иregularization_losses
╣__call__
+║&call_and_return_all_conditional_losses
'║"call_and_return_conditional_losses*

╩trace_0* 

╦trace_0* 
e_
VARIABLE_VALUEcontract_3_3x3/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEcontract_3_3x3/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
ю
╠non_trainable_variables
═layers
╬metrics
 ¤layer_regularization_losses
лlayer_metrics
Й	variables
┐trainable_variables
└regularization_losses
┬__call__
+├&call_and_return_all_conditional_losses
'├"call_and_return_conditional_losses* 

Лtrace_0* 

мtrace_0* 
* 
* 
* 
ю
Мnon_trainable_variables
нlayers
Нmetrics
 оlayer_regularization_losses
Оlayer_metrics
─	variables
┼trainable_variables
кregularization_losses
╚__call__
+╔&call_and_return_all_conditional_losses
'╔"call_and_return_conditional_losses* 

пtrace_0* 

┘trace_0* 

л0
Л1*

л0
Л1*
* 
ъ
┌non_trainable_variables
█layers
▄metrics
 Пlayer_regularization_losses
яlayer_metrics
╩	variables
╦trainable_variables
╠regularization_losses
╬__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses*

▀trace_0* 

Яtrace_0* 
c]
VARIABLE_VALUEexpand_4_5x5/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEexpand_4_5x5/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

┘0
┌1*

┘0
┌1*
* 
ъ
рnon_trainable_variables
Рlayers
сmetrics
 Сlayer_regularization_losses
тlayer_metrics
М	variables
нtrainable_variables
Нregularization_losses
О__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses*

Тtrace_0* 

уtrace_0* 
e_
VARIABLE_VALUEcontract_4_3x3/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEcontract_4_3x3/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
ю
Уnon_trainable_variables
жlayers
Жmetrics
 вlayer_regularization_losses
Вlayer_metrics
▄	variables
Пtrainable_variables
яregularization_losses
Я__call__
+р&call_and_return_all_conditional_losses
'р"call_and_return_conditional_losses* 

ьtrace_0* 

Ьtrace_0* 
* 
* 
* 
ю
№non_trainable_variables
­layers
ыmetrics
 Ыlayer_regularization_losses
зlayer_metrics
Р	variables
сtrainable_variables
Сregularization_losses
Т__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses* 

Зtrace_0* 

шtrace_0* 

Ь0
№1*

Ь0
№1*
* 
ъ
Шnon_trainable_variables
эlayers
Эmetrics
 щlayer_regularization_losses
Щlayer_metrics
У	variables
жtrainable_variables
Жregularization_losses
В__call__
+ь&call_and_return_all_conditional_losses
'ь"call_and_return_conditional_losses*

чtrace_0* 

Чtrace_0* 
d^
VARIABLE_VALUEexpand_5_5x5/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEexpand_5_5x5/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

э0
Э1*

э0
Э1*
* 
ъ
§non_trainable_variables
■layers
 metrics
 ђlayer_regularization_losses
Ђlayer_metrics
ы	variables
Ыtrainable_variables
зregularization_losses
ш__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses*

ѓtrace_0* 

Ѓtrace_0* 
f`
VARIABLE_VALUEcontract_5_3x3/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEcontract_5_3x3/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
ю
ёnon_trainable_variables
Ёlayers
єmetrics
 Єlayer_regularization_losses
ѕlayer_metrics
Щ	variables
чtrainable_variables
Чregularization_losses
■__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses* 

Ѕtrace_0* 

іtrace_0* 
* 
* 
* 
ю
Іnon_trainable_variables
їlayers
Їmetrics
 јlayer_regularization_losses
Јlayer_metrics
ђ	variables
Ђtrainable_variables
ѓregularization_losses
ё__call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses* 

љtrace_0* 

Љtrace_0* 

ї0
Ї1*

ї0
Ї1*
* 
ъ
њnon_trainable_variables
Њlayers
ћmetrics
 Ћlayer_regularization_losses
ќlayer_metrics
є	variables
Єtrainable_variables
ѕregularization_losses
і__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses*

Ќtrace_0* 

ўtrace_0* 
d^
VARIABLE_VALUEexpand_6_5x5/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEexpand_6_5x5/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

Ћ0
ќ1*

Ћ0
ќ1*
* 
ъ
Ўnon_trainable_variables
џlayers
Џmetrics
 юlayer_regularization_losses
Юlayer_metrics
Ј	variables
љtrainable_variables
Љregularization_losses
Њ__call__
+ћ&call_and_return_all_conditional_losses
'ћ"call_and_return_conditional_losses*

ъtrace_0* 

Ъtrace_0* 
f`
VARIABLE_VALUEcontract_6_3x3/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEcontract_6_3x3/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
ю
аnon_trainable_variables
Аlayers
бmetrics
 Бlayer_regularization_losses
цlayer_metrics
ў	variables
Ўtrainable_variables
џregularization_losses
ю__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses* 

Цtrace_0* 

дtrace_0* 
* 
* 
* 
ю
Дnon_trainable_variables
еlayers
Еmetrics
 фlayer_regularization_losses
Фlayer_metrics
ъ	variables
Ъtrainable_variables
аregularization_losses
б__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses* 

гtrace_0* 

Гtrace_0* 

ф0
Ф1*

ф0
Ф1*
* 
ъ
«non_trainable_variables
»layers
░metrics
 ▒layer_regularization_losses
▓layer_metrics
ц	variables
Цtrainable_variables
дregularization_losses
е__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses*

│trace_0* 

┤trace_0* 
d^
VARIABLE_VALUEexpand_7_5x5/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEexpand_7_5x5/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

│0
┤1*

│0
┤1*
* 
ъ
хnon_trainable_variables
Хlayers
иmetrics
 Иlayer_regularization_losses
╣layer_metrics
Г	variables
«trainable_variables
»regularization_losses
▒__call__
+▓&call_and_return_all_conditional_losses
'▓"call_and_return_conditional_losses*

║trace_0* 

╗trace_0* 
f`
VARIABLE_VALUEcontract_7_3x3/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEcontract_7_3x3/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
ю
╝non_trainable_variables
йlayers
Йmetrics
 ┐layer_regularization_losses
└layer_metrics
Х	variables
иtrainable_variables
Иregularization_losses
║__call__
+╗&call_and_return_all_conditional_losses
'╗"call_and_return_conditional_losses* 

┴trace_0* 

┬trace_0* 
* 
* 
* 
ю
├non_trainable_variables
─layers
┼metrics
 кlayer_regularization_losses
Кlayer_metrics
╝	variables
йtrainable_variables
Йregularization_losses
└__call__
+┴&call_and_return_all_conditional_losses
'┴"call_and_return_conditional_losses* 

╚trace_0* 

╔trace_0* 

╚0
╔1*

╚0
╔1*
* 
ъ
╩non_trainable_variables
╦layers
╠metrics
 ═layer_regularization_losses
╬layer_metrics
┬	variables
├trainable_variables
─regularization_losses
к__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses*

¤trace_0* 

лtrace_0* 
d^
VARIABLE_VALUEexpand_8_5x5/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEexpand_8_5x5/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

Л0
м1*

Л0
м1*
* 
ъ
Лnon_trainable_variables
мlayers
Мmetrics
 нlayer_regularization_losses
Нlayer_metrics
╦	variables
╠trainable_variables
═regularization_losses
¤__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses*

оtrace_0* 

Оtrace_0* 
f`
VARIABLE_VALUEcontract_8_3x3/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEcontract_8_3x3/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
ю
пnon_trainable_variables
┘layers
┌metrics
 █layer_regularization_losses
▄layer_metrics
н	variables
Нtrainable_variables
оregularization_losses
п__call__
+┘&call_and_return_all_conditional_losses
'┘"call_and_return_conditional_losses* 

Пtrace_0* 

яtrace_0* 
* 
* 
* 
ю
▀non_trainable_variables
Яlayers
рmetrics
 Рlayer_regularization_losses
сlayer_metrics
┌	variables
█trainable_variables
▄regularization_losses
я__call__
+▀&call_and_return_all_conditional_losses
'▀"call_and_return_conditional_losses* 

Сtrace_0* 

тtrace_0* 

Т0
у1*

Т0
у1*
* 
ъ
Тnon_trainable_variables
уlayers
Уmetrics
 жlayer_regularization_losses
Жlayer_metrics
Я	variables
рtrainable_variables
Рregularization_losses
С__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses*

вtrace_0* 

Вtrace_0* 
d^
VARIABLE_VALUEexpand_9_5x5/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEexpand_9_5x5/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

№0
­1*

№0
­1*
* 
ъ
ьnon_trainable_variables
Ьlayers
№metrics
 ­layer_regularization_losses
ыlayer_metrics
ж	variables
Жtrainable_variables
вregularization_losses
ь__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses*

Ыtrace_0* 

зtrace_0* 
f`
VARIABLE_VALUEcontract_9_3x3/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEcontract_9_3x3/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
ю
Зnon_trainable_variables
шlayers
Шmetrics
 эlayer_regularization_losses
Эlayer_metrics
Ы	variables
зtrainable_variables
Зregularization_losses
Ш__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses* 

щtrace_0* 

Щtrace_0* 
* 
* 
* 
ю
чnon_trainable_variables
Чlayers
§metrics
 ■layer_regularization_losses
 layer_metrics
Э	variables
щtrainable_variables
Щregularization_losses
Ч__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses* 

ђtrace_0* 

Ђtrace_0* 

ё0
Ё1*

ё0
Ё1*
* 
ъ
ѓnon_trainable_variables
Ѓlayers
ёmetrics
 Ёlayer_regularization_losses
єlayer_metrics
■	variables
 trainable_variables
ђregularization_losses
ѓ__call__
+Ѓ&call_and_return_all_conditional_losses
'Ѓ"call_and_return_conditional_losses*

Єtrace_0* 

ѕtrace_0* 
e_
VARIABLE_VALUEexpand_10_5x5/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEexpand_10_5x5/bias5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

Ї0
ј1*

Ї0
ј1*
* 
ъ
Ѕnon_trainable_variables
іlayers
Іmetrics
 їlayer_regularization_losses
Їlayer_metrics
Є	variables
ѕtrainable_variables
Ѕregularization_losses
І__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses*

јtrace_0* 

Јtrace_0* 
ga
VARIABLE_VALUEcontract_10_3x3/kernel7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEcontract_10_3x3/bias5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
ю
љnon_trainable_variables
Љlayers
њmetrics
 Њlayer_regularization_losses
ћlayer_metrics
љ	variables
Љtrainable_variables
њregularization_losses
ћ__call__
+Ћ&call_and_return_all_conditional_losses
'Ћ"call_and_return_conditional_losses* 

Ћtrace_0* 

ќtrace_0* 
* 
* 
* 
ю
Ќnon_trainable_variables
ўlayers
Ўmetrics
 џlayer_regularization_losses
Џlayer_metrics
ќ	variables
Ќtrainable_variables
ўregularization_losses
џ__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses* 

юtrace_0* 

Юtrace_0* 

б0
Б1*

б0
Б1*
* 
ъ
ъnon_trainable_variables
Ъlayers
аmetrics
 Аlayer_regularization_losses
бlayer_metrics
ю	variables
Юtrainable_variables
ъregularization_losses
а__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses*

Бtrace_0* 

цtrace_0* 
e_
VARIABLE_VALUEexpand_11_5x5/kernel7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEexpand_11_5x5/bias5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

Ф0
г1*

Ф0
г1*
* 
ъ
Цnon_trainable_variables
дlayers
Дmetrics
 еlayer_regularization_losses
Еlayer_metrics
Ц	variables
дtrainable_variables
Дregularization_losses
Е__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses*

фtrace_0* 

Фtrace_0* 
ga
VARIABLE_VALUEcontract_11_3x3/kernel7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEcontract_11_3x3/bias5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
ю
гnon_trainable_variables
Гlayers
«metrics
 »layer_regularization_losses
░layer_metrics
«	variables
»trainable_variables
░regularization_losses
▓__call__
+│&call_and_return_all_conditional_losses
'│"call_and_return_conditional_losses* 

▒trace_0* 

▓trace_0* 
* 
* 
* 
ю
│non_trainable_variables
┤layers
хmetrics
 Хlayer_regularization_losses
иlayer_metrics
┤	variables
хtrainable_variables
Хregularization_losses
И__call__
+╣&call_and_return_all_conditional_losses
'╣"call_and_return_conditional_losses* 

Иtrace_0* 

╣trace_0* 

└0
┴1*

└0
┴1*
* 
ъ
║non_trainable_variables
╗layers
╝metrics
 йlayer_regularization_losses
Йlayer_metrics
║	variables
╗trainable_variables
╝regularization_losses
Й__call__
+┐&call_and_return_all_conditional_losses
'┐"call_and_return_conditional_losses*

┐trace_0* 

└trace_0* 
e_
VARIABLE_VALUEexpand_12_5x5/kernel7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEexpand_12_5x5/bias5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

╔0
╩1*

╔0
╩1*
* 
ъ
┴non_trainable_variables
┬layers
├metrics
 ─layer_regularization_losses
┼layer_metrics
├	variables
─trainable_variables
┼regularization_losses
К__call__
+╚&call_and_return_all_conditional_losses
'╚"call_and_return_conditional_losses*

кtrace_0* 

Кtrace_0* 
ga
VARIABLE_VALUEcontract_12_3x3/kernel7layer_with_weights-25/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEcontract_12_3x3/bias5layer_with_weights-25/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
ю
╚non_trainable_variables
╔layers
╩metrics
 ╦layer_regularization_losses
╠layer_metrics
╠	variables
═trainable_variables
╬regularization_losses
л__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses* 

═trace_0* 

╬trace_0* 
* 
* 
* 
ю
¤non_trainable_variables
лlayers
Лmetrics
 мlayer_regularization_losses
Мlayer_metrics
м	variables
Мtrainable_variables
нregularization_losses
о__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses* 

нtrace_0* 

Нtrace_0* 

я0
▀1*

я0
▀1*
* 
ъ
оnon_trainable_variables
Оlayers
пmetrics
 ┘layer_regularization_losses
┌layer_metrics
п	variables
┘trainable_variables
┌regularization_losses
▄__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses*

█trace_0* 

▄trace_0* 
e_
VARIABLE_VALUEexpand_13_5x5/kernel7layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEexpand_13_5x5/bias5layer_with_weights-26/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

у0
У1*

у0
У1*
* 
ъ
Пnon_trainable_variables
яlayers
▀metrics
 Яlayer_regularization_losses
рlayer_metrics
р	variables
Рtrainable_variables
сregularization_losses
т__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses*

Рtrace_0* 

сtrace_0* 
ga
VARIABLE_VALUEcontract_13_3x3/kernel7layer_with_weights-27/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEcontract_13_3x3/bias5layer_with_weights-27/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
ю
Сnon_trainable_variables
тlayers
Тmetrics
 уlayer_regularization_losses
Уlayer_metrics
Ж	variables
вtrainable_variables
Вregularization_losses
Ь__call__
+№&call_and_return_all_conditional_losses
'№"call_and_return_conditional_losses* 

жtrace_0* 

Жtrace_0* 
* 
* 
* 
ю
вnon_trainable_variables
Вlayers
ьmetrics
 Ьlayer_regularization_losses
№layer_metrics
­	variables
ыtrainable_variables
Ыregularization_losses
З__call__
+ш&call_and_return_all_conditional_losses
'ш"call_and_return_conditional_losses* 

­trace_0* 

ыtrace_0* 

Ч0
§1*

Ч0
§1*
* 
ъ
Ыnon_trainable_variables
зlayers
Зmetrics
 шlayer_regularization_losses
Шlayer_metrics
Ш	variables
эtrainable_variables
Эregularization_losses
Щ__call__
+ч&call_and_return_all_conditional_losses
'ч"call_and_return_conditional_losses*

эtrace_0* 

Эtrace_0* 
e_
VARIABLE_VALUEexpand_14_5x5/kernel7layer_with_weights-28/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEexpand_14_5x5/bias5layer_with_weights-28/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

Ё0
є1*

Ё0
є1*
* 
ъ
щnon_trainable_variables
Щlayers
чmetrics
 Чlayer_regularization_losses
§layer_metrics
 	variables
ђtrainable_variables
Ђregularization_losses
Ѓ__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses*

■trace_0* 

 trace_0* 
ga
VARIABLE_VALUEcontract_14_3x3/kernel7layer_with_weights-29/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEcontract_14_3x3/bias5layer_with_weights-29/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
ю
ђ	non_trainable_variables
Ђ	layers
ѓ	metrics
 Ѓ	layer_regularization_losses
ё	layer_metrics
ѕ	variables
Ѕtrainable_variables
іregularization_losses
ї__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses* 

Ё	trace_0* 

є	trace_0* 
* 
* 
* 
ю
Є	non_trainable_variables
ѕ	layers
Ѕ	metrics
 і	layer_regularization_losses
І	layer_metrics
ј	variables
Јtrainable_variables
љregularization_losses
њ__call__
+Њ&call_and_return_all_conditional_losses
'Њ"call_and_return_conditional_losses* 

ї	trace_0* 

Ї	trace_0* 

џ0
Џ1*

џ0
Џ1*
* 
ъ
ј	non_trainable_variables
Ј	layers
љ	metrics
 Љ	layer_regularization_losses
њ	layer_metrics
ћ	variables
Ћtrainable_variables
ќregularization_losses
ў__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses*

Њ	trace_0* 

ћ	trace_0* 
e_
VARIABLE_VALUEexpand_15_5x5/kernel7layer_with_weights-30/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEexpand_15_5x5/bias5layer_with_weights-30/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

Б0
ц1*

Б0
ц1*
* 
ъ
Ћ	non_trainable_variables
ќ	layers
Ќ	metrics
 ў	layer_regularization_losses
Ў	layer_metrics
Ю	variables
ъtrainable_variables
Ъregularization_losses
А__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses*

џ	trace_0* 

Џ	trace_0* 
ga
VARIABLE_VALUEcontract_15_3x3/kernel7layer_with_weights-31/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEcontract_15_3x3/bias5layer_with_weights-31/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
ю
ю	non_trainable_variables
Ю	layers
ъ	metrics
 Ъ	layer_regularization_losses
а	layer_metrics
д	variables
Дtrainable_variables
еregularization_losses
ф__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses* 

А	trace_0* 

б	trace_0* 
* 
* 
* 
ю
Б	non_trainable_variables
ц	layers
Ц	metrics
 д	layer_regularization_losses
Д	layer_metrics
г	variables
Гtrainable_variables
«regularization_losses
░__call__
+▒&call_and_return_all_conditional_losses
'▒"call_and_return_conditional_losses* 

е	trace_0* 

Е	trace_0* 

И0
╣1*

И0
╣1*
* 
ъ
ф	non_trainable_variables
Ф	layers
г	metrics
 Г	layer_regularization_losses
«	layer_metrics
▓	variables
│trainable_variables
┤regularization_losses
Х__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses*

»	trace_0* 

░	trace_0* 
e_
VARIABLE_VALUEexpand_16_5x5/kernel7layer_with_weights-32/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEexpand_16_5x5/bias5layer_with_weights-32/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

┴0
┬1*

┴0
┬1*
* 
ъ
▒	non_trainable_variables
▓	layers
│	metrics
 ┤	layer_regularization_losses
х	layer_metrics
╗	variables
╝trainable_variables
йregularization_losses
┐__call__
+└&call_and_return_all_conditional_losses
'└"call_and_return_conditional_losses*

Х	trace_0* 

и	trace_0* 
ga
VARIABLE_VALUEcontract_16_3x3/kernel7layer_with_weights-33/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEcontract_16_3x3/bias5layer_with_weights-33/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
ю
И	non_trainable_variables
╣	layers
║	metrics
 ╗	layer_regularization_losses
╝	layer_metrics
─	variables
┼trainable_variables
кregularization_losses
╚__call__
+╔&call_and_return_all_conditional_losses
'╔"call_and_return_conditional_losses* 

й	trace_0* 

Й	trace_0* 
* 
* 
* 
ю
┐	non_trainable_variables
└	layers
┴	metrics
 ┬	layer_regularization_losses
├	layer_metrics
╩	variables
╦trainable_variables
╠regularization_losses
╬__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses* 

─	trace_0* 

┼	trace_0* 

о0
О1*

о0
О1*
* 
ъ
к	non_trainable_variables
К	layers
╚	metrics
 ╔	layer_regularization_losses
╩	layer_metrics
л	variables
Лtrainable_variables
мregularization_losses
н__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses*

╦	trace_0* 

╠	trace_0* 
e_
VARIABLE_VALUEexpand_17_5x5/kernel7layer_with_weights-34/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEexpand_17_5x5/bias5layer_with_weights-34/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

▀0
Я1*

▀0
Я1*
* 
ъ
═	non_trainable_variables
╬	layers
¤	metrics
 л	layer_regularization_losses
Л	layer_metrics
┘	variables
┌trainable_variables
█regularization_losses
П__call__
+я&call_and_return_all_conditional_losses
'я"call_and_return_conditional_losses*

м	trace_0* 

М	trace_0* 
ga
VARIABLE_VALUEcontract_17_3x3/kernel7layer_with_weights-35/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEcontract_17_3x3/bias5layer_with_weights-35/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
ю
н	non_trainable_variables
Н	layers
о	metrics
 О	layer_regularization_losses
п	layer_metrics
Р	variables
сtrainable_variables
Сregularization_losses
Т__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses* 

┘	trace_0* 

┌	trace_0* 
* 
* 
* 
ю
█	non_trainable_variables
▄	layers
П	metrics
 я	layer_regularization_losses
▀	layer_metrics
У	variables
жtrainable_variables
Жregularization_losses
В__call__
+ь&call_and_return_all_conditional_losses
'ь"call_and_return_conditional_losses* 

Я	trace_0* 

р	trace_0* 

З0
ш1*

З0
ш1*
* 
ъ
Р	non_trainable_variables
с	layers
С	metrics
 т	layer_regularization_losses
Т	layer_metrics
Ь	variables
№trainable_variables
­regularization_losses
Ы__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses*

у	trace_0* 

У	trace_0* 
e_
VARIABLE_VALUEexpand_18_5x5/kernel7layer_with_weights-36/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEexpand_18_5x5/bias5layer_with_weights-36/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

§0
■1*

§0
■1*
* 
ъ
ж	non_trainable_variables
Ж	layers
в	metrics
 В	layer_regularization_losses
ь	layer_metrics
э	variables
Эtrainable_variables
щregularization_losses
ч__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses*

Ь	trace_0* 

№	trace_0* 
ga
VARIABLE_VALUEcontract_18_3x3/kernel7layer_with_weights-37/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEcontract_18_3x3/bias5layer_with_weights-37/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
ю
­	non_trainable_variables
ы	layers
Ы	metrics
 з	layer_regularization_losses
З	layer_metrics
ђ	variables
Ђtrainable_variables
ѓregularization_losses
ё__call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses* 

ш	trace_0* 

Ш	trace_0* 
* 
* 
* 
ю
э	non_trainable_variables
Э	layers
щ	metrics
 Щ	layer_regularization_losses
ч	layer_metrics
є	variables
Єtrainable_variables
ѕregularization_losses
і__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses* 

Ч	trace_0* 

§	trace_0* 

њ0
Њ1*

њ0
Њ1*
* 
ъ
■	non_trainable_variables
 	layers
ђ
metrics
 Ђ
layer_regularization_losses
ѓ
layer_metrics
ї	variables
Їtrainable_variables
јregularization_losses
љ__call__
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses*

Ѓ
trace_0* 

ё
trace_0* 
e_
VARIABLE_VALUEexpand_19_5x5/kernel7layer_with_weights-38/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEexpand_19_5x5/bias5layer_with_weights-38/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

Џ0
ю1*

Џ0
ю1*
* 
ъ
Ё
non_trainable_variables
є
layers
Є
metrics
 ѕ
layer_regularization_losses
Ѕ
layer_metrics
Ћ	variables
ќtrainable_variables
Ќregularization_losses
Ў__call__
+џ&call_and_return_all_conditional_losses
'џ"call_and_return_conditional_losses*

і
trace_0* 

І
trace_0* 
ga
VARIABLE_VALUEcontract_19_3x3/kernel7layer_with_weights-39/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEcontract_19_3x3/bias5layer_with_weights-39/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
ю
ї
non_trainable_variables
Ї
layers
ј
metrics
 Ј
layer_regularization_losses
љ
layer_metrics
ъ	variables
Ъtrainable_variables
аregularization_losses
б__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses* 

Љ
trace_0* 

њ
trace_0* 
* 
* 
* 
ю
Њ
non_trainable_variables
ћ
layers
Ћ
metrics
 ќ
layer_regularization_losses
Ќ
layer_metrics
ц	variables
Цtrainable_variables
дregularization_losses
е__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses* 

ў
trace_0* 

Ў
trace_0* 

░0
▒1*

░0
▒1*
* 
ъ
џ
non_trainable_variables
Џ
layers
ю
metrics
 Ю
layer_regularization_losses
ъ
layer_metrics
ф	variables
Фtrainable_variables
гregularization_losses
«__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses*

Ъ
trace_0* 

а
trace_0* 
e_
VARIABLE_VALUEexpand_20_5x5/kernel7layer_with_weights-40/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEexpand_20_5x5/bias5layer_with_weights-40/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

╣0
║1*

╣0
║1*
* 
ъ
А
non_trainable_variables
б
layers
Б
metrics
 ц
layer_regularization_losses
Ц
layer_metrics
│	variables
┤trainable_variables
хregularization_losses
и__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses*

д
trace_0* 

Д
trace_0* 
ga
VARIABLE_VALUEcontract_20_3x3/kernel7layer_with_weights-41/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEcontract_20_3x3/bias5layer_with_weights-41/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
ю
е
non_trainable_variables
Е
layers
ф
metrics
 Ф
layer_regularization_losses
г
layer_metrics
╝	variables
йtrainable_variables
Йregularization_losses
└__call__
+┴&call_and_return_all_conditional_losses
'┴"call_and_return_conditional_losses* 

Г
trace_0* 

«
trace_0* 

╚0
╔1*

╚0
╔1*
* 
ъ
»
non_trainable_variables
░
layers
▒
metrics
 ▓
layer_regularization_losses
│
layer_metrics
┬	variables
├trainable_variables
─regularization_losses
к__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses*

┤
trace_0* 

х
trace_0* 
ic
VARIABLE_VALUEpolicy_aggregator/kernel7layer_with_weights-42/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEpolicy_aggregator/bias5layer_with_weights-42/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
ю
Х
non_trainable_variables
и
layers
И
metrics
 ╣
layer_regularization_losses
║
layer_metrics
╦	variables
╠trainable_variables
═regularization_losses
¤__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses* 

╗
trace_0* 

╝
trace_0* 

О0
п1*
* 
* 
ъ
й
non_trainable_variables
Й
layers
┐
metrics
 └
layer_regularization_losses
┴
layer_metrics
Л	variables
мtrainable_variables
Мregularization_losses
Н__call__
+о&call_and_return_all_conditional_losses
'о"call_and_return_conditional_losses*

┬
trace_0* 

├
trace_0* 
b\
VARIABLE_VALUEborder_off/kernel7layer_with_weights-43/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEborder_off/bias5layer_with_weights-43/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
ю
─
non_trainable_variables
┼
layers
к
metrics
 К
layer_regularization_losses
╚
layer_metrics
┌	variables
█trainable_variables
▄regularization_losses
я__call__
+▀&call_and_return_all_conditional_losses
'▀"call_and_return_conditional_losses* 

╔
trace_0* 

╩
trace_0* 
* 
* 
* 
ю
╦
non_trainable_variables
╠
layers
═
metrics
 ╬
layer_regularization_losses
¤
layer_metrics
Я	variables
рtrainable_variables
Рregularization_losses
С__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses* 

л
trace_0* 

Л
trace_0* 
* 
* 
* 
* 
ю
м
non_trainable_variables
М
layers
н
metrics
 Н
layer_regularization_losses
о
layer_metrics
у	variables
Уtrainable_variables
жregularization_losses
в__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses* 

О
trace_0* 

п
trace_0* 

з0
З1*

з0
З1*
* 
ъ
┘
non_trainable_variables
┌
layers
█
metrics
 ▄
layer_regularization_losses
П
layer_metrics
ь	variables
Ьtrainable_variables
№regularization_losses
ы__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses*

я
trace_0* 

▀
trace_0* 
b\
VARIABLE_VALUEvalue_head/kernel7layer_with_weights-44/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEvalue_head/bias5layer_with_weights-44/bias/.ATTRIBUTES/VARIABLE_VALUE*
0
j0
k1
|2
}3
О4
п5*
┬
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44
.45
/46
047
148
249
350
451
552
653
754
855
956
:57
;58
<59
=60
>61
?62
@63
A64
B65
C66
D67
E68
F69
G70
H71
I72
J73
K74
L75
M76
N77
O78
P79
Q80
R81
S82
T83
U84
V85
W86
X87
Y88*

Я
0*
* 
* 
* 
* 
* 
* 
* 

j0
k1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

|0
}1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

О0
п1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
р
	variables
Р
	keras_api

с
total

С
count*

с
0
С
1*

р
	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
Ѕ
serving_default_inputsPlaceholder*/
_output_shapes
:         *
dtype0*$
shape:         
А
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputsexpand_1_11x11/kernelexpand_1_11x11/biasheuristic_detector/kernelheuristic_detector/biasheuristic_priority/kernelheuristic_priority/biascontract_1_5x5/kernelcontract_1_5x5/biasexpand_2_5x5/kernelexpand_2_5x5/biascontract_2_3x3/kernelcontract_2_3x3/biasexpand_3_5x5/kernelexpand_3_5x5/biascontract_3_3x3/kernelcontract_3_3x3/biasexpand_4_5x5/kernelexpand_4_5x5/biascontract_4_3x3/kernelcontract_4_3x3/biasexpand_5_5x5/kernelexpand_5_5x5/biascontract_5_3x3/kernelcontract_5_3x3/biasexpand_6_5x5/kernelexpand_6_5x5/biascontract_6_3x3/kernelcontract_6_3x3/biasexpand_7_5x5/kernelexpand_7_5x5/biascontract_7_3x3/kernelcontract_7_3x3/biasexpand_8_5x5/kernelexpand_8_5x5/biascontract_8_3x3/kernelcontract_8_3x3/biasexpand_9_5x5/kernelexpand_9_5x5/biascontract_9_3x3/kernelcontract_9_3x3/biasexpand_10_5x5/kernelexpand_10_5x5/biascontract_10_3x3/kernelcontract_10_3x3/biasexpand_11_5x5/kernelexpand_11_5x5/biascontract_11_3x3/kernelcontract_11_3x3/biasexpand_12_5x5/kernelexpand_12_5x5/biascontract_12_3x3/kernelcontract_12_3x3/biasexpand_13_5x5/kernelexpand_13_5x5/biascontract_13_3x3/kernelcontract_13_3x3/biasexpand_14_5x5/kernelexpand_14_5x5/biascontract_14_3x3/kernelcontract_14_3x3/biasexpand_15_5x5/kernelexpand_15_5x5/biascontract_15_3x3/kernelcontract_15_3x3/biasexpand_16_5x5/kernelexpand_16_5x5/biascontract_16_3x3/kernelcontract_16_3x3/biasexpand_17_5x5/kernelexpand_17_5x5/biascontract_17_3x3/kernelcontract_17_3x3/biasexpand_18_5x5/kernelexpand_18_5x5/biascontract_18_3x3/kernelcontract_18_3x3/biasexpand_19_5x5/kernelexpand_19_5x5/biascontract_19_3x3/kernelcontract_19_3x3/biasexpand_20_5x5/kernelexpand_20_5x5/biascontract_20_3x3/kernelcontract_20_3x3/biaspolicy_aggregator/kernelpolicy_aggregator/biasborder_off/kernelborder_off/biasvalue_head/kernelvalue_head/bias*f
Tin_
]2[*
Tout
2*
_collective_manager_ids
 *;
_output_shapes)
':         ж:         *|
_read_only_resource_inputs^
\Z	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ*0
config_proto 

CPU

GPU2*0J 8ѓ */
f*R(
&__inference_signature_wrapper_10432559
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
▄ 
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-heuristic_detector/kernel/Read/ReadVariableOp+heuristic_detector/bias/Read/ReadVariableOp)expand_1_11x11/kernel/Read/ReadVariableOp'expand_1_11x11/bias/Read/ReadVariableOp-heuristic_priority/kernel/Read/ReadVariableOp+heuristic_priority/bias/Read/ReadVariableOp)contract_1_5x5/kernel/Read/ReadVariableOp'contract_1_5x5/bias/Read/ReadVariableOp'expand_2_5x5/kernel/Read/ReadVariableOp%expand_2_5x5/bias/Read/ReadVariableOp)contract_2_3x3/kernel/Read/ReadVariableOp'contract_2_3x3/bias/Read/ReadVariableOp'expand_3_5x5/kernel/Read/ReadVariableOp%expand_3_5x5/bias/Read/ReadVariableOp)contract_3_3x3/kernel/Read/ReadVariableOp'contract_3_3x3/bias/Read/ReadVariableOp'expand_4_5x5/kernel/Read/ReadVariableOp%expand_4_5x5/bias/Read/ReadVariableOp)contract_4_3x3/kernel/Read/ReadVariableOp'contract_4_3x3/bias/Read/ReadVariableOp'expand_5_5x5/kernel/Read/ReadVariableOp%expand_5_5x5/bias/Read/ReadVariableOp)contract_5_3x3/kernel/Read/ReadVariableOp'contract_5_3x3/bias/Read/ReadVariableOp'expand_6_5x5/kernel/Read/ReadVariableOp%expand_6_5x5/bias/Read/ReadVariableOp)contract_6_3x3/kernel/Read/ReadVariableOp'contract_6_3x3/bias/Read/ReadVariableOp'expand_7_5x5/kernel/Read/ReadVariableOp%expand_7_5x5/bias/Read/ReadVariableOp)contract_7_3x3/kernel/Read/ReadVariableOp'contract_7_3x3/bias/Read/ReadVariableOp'expand_8_5x5/kernel/Read/ReadVariableOp%expand_8_5x5/bias/Read/ReadVariableOp)contract_8_3x3/kernel/Read/ReadVariableOp'contract_8_3x3/bias/Read/ReadVariableOp'expand_9_5x5/kernel/Read/ReadVariableOp%expand_9_5x5/bias/Read/ReadVariableOp)contract_9_3x3/kernel/Read/ReadVariableOp'contract_9_3x3/bias/Read/ReadVariableOp(expand_10_5x5/kernel/Read/ReadVariableOp&expand_10_5x5/bias/Read/ReadVariableOp*contract_10_3x3/kernel/Read/ReadVariableOp(contract_10_3x3/bias/Read/ReadVariableOp(expand_11_5x5/kernel/Read/ReadVariableOp&expand_11_5x5/bias/Read/ReadVariableOp*contract_11_3x3/kernel/Read/ReadVariableOp(contract_11_3x3/bias/Read/ReadVariableOp(expand_12_5x5/kernel/Read/ReadVariableOp&expand_12_5x5/bias/Read/ReadVariableOp*contract_12_3x3/kernel/Read/ReadVariableOp(contract_12_3x3/bias/Read/ReadVariableOp(expand_13_5x5/kernel/Read/ReadVariableOp&expand_13_5x5/bias/Read/ReadVariableOp*contract_13_3x3/kernel/Read/ReadVariableOp(contract_13_3x3/bias/Read/ReadVariableOp(expand_14_5x5/kernel/Read/ReadVariableOp&expand_14_5x5/bias/Read/ReadVariableOp*contract_14_3x3/kernel/Read/ReadVariableOp(contract_14_3x3/bias/Read/ReadVariableOp(expand_15_5x5/kernel/Read/ReadVariableOp&expand_15_5x5/bias/Read/ReadVariableOp*contract_15_3x3/kernel/Read/ReadVariableOp(contract_15_3x3/bias/Read/ReadVariableOp(expand_16_5x5/kernel/Read/ReadVariableOp&expand_16_5x5/bias/Read/ReadVariableOp*contract_16_3x3/kernel/Read/ReadVariableOp(contract_16_3x3/bias/Read/ReadVariableOp(expand_17_5x5/kernel/Read/ReadVariableOp&expand_17_5x5/bias/Read/ReadVariableOp*contract_17_3x3/kernel/Read/ReadVariableOp(contract_17_3x3/bias/Read/ReadVariableOp(expand_18_5x5/kernel/Read/ReadVariableOp&expand_18_5x5/bias/Read/ReadVariableOp*contract_18_3x3/kernel/Read/ReadVariableOp(contract_18_3x3/bias/Read/ReadVariableOp(expand_19_5x5/kernel/Read/ReadVariableOp&expand_19_5x5/bias/Read/ReadVariableOp*contract_19_3x3/kernel/Read/ReadVariableOp(contract_19_3x3/bias/Read/ReadVariableOp(expand_20_5x5/kernel/Read/ReadVariableOp&expand_20_5x5/bias/Read/ReadVariableOp*contract_20_3x3/kernel/Read/ReadVariableOp(contract_20_3x3/bias/Read/ReadVariableOp,policy_aggregator/kernel/Read/ReadVariableOp*policy_aggregator/bias/Read/ReadVariableOp%border_off/kernel/Read/ReadVariableOp#border_off/bias/Read/ReadVariableOp%value_head/kernel/Read/ReadVariableOp#value_head/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*i
Tinb
`2^*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ **
f%R#
!__inference__traced_save_10434278
Д
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameheuristic_detector/kernelheuristic_detector/biasexpand_1_11x11/kernelexpand_1_11x11/biasheuristic_priority/kernelheuristic_priority/biascontract_1_5x5/kernelcontract_1_5x5/biasexpand_2_5x5/kernelexpand_2_5x5/biascontract_2_3x3/kernelcontract_2_3x3/biasexpand_3_5x5/kernelexpand_3_5x5/biascontract_3_3x3/kernelcontract_3_3x3/biasexpand_4_5x5/kernelexpand_4_5x5/biascontract_4_3x3/kernelcontract_4_3x3/biasexpand_5_5x5/kernelexpand_5_5x5/biascontract_5_3x3/kernelcontract_5_3x3/biasexpand_6_5x5/kernelexpand_6_5x5/biascontract_6_3x3/kernelcontract_6_3x3/biasexpand_7_5x5/kernelexpand_7_5x5/biascontract_7_3x3/kernelcontract_7_3x3/biasexpand_8_5x5/kernelexpand_8_5x5/biascontract_8_3x3/kernelcontract_8_3x3/biasexpand_9_5x5/kernelexpand_9_5x5/biascontract_9_3x3/kernelcontract_9_3x3/biasexpand_10_5x5/kernelexpand_10_5x5/biascontract_10_3x3/kernelcontract_10_3x3/biasexpand_11_5x5/kernelexpand_11_5x5/biascontract_11_3x3/kernelcontract_11_3x3/biasexpand_12_5x5/kernelexpand_12_5x5/biascontract_12_3x3/kernelcontract_12_3x3/biasexpand_13_5x5/kernelexpand_13_5x5/biascontract_13_3x3/kernelcontract_13_3x3/biasexpand_14_5x5/kernelexpand_14_5x5/biascontract_14_3x3/kernelcontract_14_3x3/biasexpand_15_5x5/kernelexpand_15_5x5/biascontract_15_3x3/kernelcontract_15_3x3/biasexpand_16_5x5/kernelexpand_16_5x5/biascontract_16_3x3/kernelcontract_16_3x3/biasexpand_17_5x5/kernelexpand_17_5x5/biascontract_17_3x3/kernelcontract_17_3x3/biasexpand_18_5x5/kernelexpand_18_5x5/biascontract_18_3x3/kernelcontract_18_3x3/biasexpand_19_5x5/kernelexpand_19_5x5/biascontract_19_3x3/kernelcontract_19_3x3/biasexpand_20_5x5/kernelexpand_20_5x5/biascontract_20_3x3/kernelcontract_20_3x3/biaspolicy_aggregator/kernelpolicy_aggregator/biasborder_off/kernelborder_off/biasvalue_head/kernelvalue_head/biastotalcount*h
Tina
_2]*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *-
f(R&
$__inference__traced_restore_10434564ар&
і
і
P__inference_heuristic_priority_layer_call_and_return_conditional_losses_10428983

inputs9
conv2d_readvariableop_resource:│-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:│*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         X
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:         _
IdentityIdentityTanh:y:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         │: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         │
 
_user_specified_nameinputs
Ћ
Ѓ
J__inference_expand_9_5x5_layer_call_and_return_conditional_losses_10433127

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:          m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
Їё
█.
M__inference_gomoku_resnet_1_layer_call_and_return_conditional_losses_10431450

inputs2
expand_1_11x11_10431179:ђ&
expand_1_11x11_10431181:	ђ6
heuristic_detector_10431184:│*
heuristic_detector_10431186:	│6
heuristic_priority_10431189:│)
heuristic_priority_10431191:2
contract_1_5x5_10431194:ђ%
contract_1_5x5_10431196:/
expand_2_5x5_10431200:	 #
expand_2_5x5_10431202: 1
contract_2_3x3_10431205: %
contract_2_3x3_10431207:/
expand_3_5x5_10431212:	 #
expand_3_5x5_10431214: 1
contract_3_3x3_10431217: %
contract_3_3x3_10431219:/
expand_4_5x5_10431224:	 #
expand_4_5x5_10431226: 1
contract_4_3x3_10431229: %
contract_4_3x3_10431231:/
expand_5_5x5_10431236:	 #
expand_5_5x5_10431238: 1
contract_5_3x3_10431241: %
contract_5_3x3_10431243:/
expand_6_5x5_10431248:	 #
expand_6_5x5_10431250: 1
contract_6_3x3_10431253: %
contract_6_3x3_10431255:/
expand_7_5x5_10431260:	 #
expand_7_5x5_10431262: 1
contract_7_3x3_10431265: %
contract_7_3x3_10431267:/
expand_8_5x5_10431272:	 #
expand_8_5x5_10431274: 1
contract_8_3x3_10431277: %
contract_8_3x3_10431279:/
expand_9_5x5_10431284:	 #
expand_9_5x5_10431286: 1
contract_9_3x3_10431289: %
contract_9_3x3_10431291:0
expand_10_5x5_10431296:	 $
expand_10_5x5_10431298: 2
contract_10_3x3_10431301: &
contract_10_3x3_10431303:0
expand_11_5x5_10431308:	 $
expand_11_5x5_10431310: 2
contract_11_3x3_10431313: &
contract_11_3x3_10431315:0
expand_12_5x5_10431320:	 $
expand_12_5x5_10431322: 2
contract_12_3x3_10431325: &
contract_12_3x3_10431327:0
expand_13_5x5_10431332:	 $
expand_13_5x5_10431334: 2
contract_13_3x3_10431337: &
contract_13_3x3_10431339:0
expand_14_5x5_10431344:	 $
expand_14_5x5_10431346: 2
contract_14_3x3_10431349: &
contract_14_3x3_10431351:0
expand_15_5x5_10431356:	 $
expand_15_5x5_10431358: 2
contract_15_3x3_10431361: &
contract_15_3x3_10431363:0
expand_16_5x5_10431368:	 $
expand_16_5x5_10431370: 2
contract_16_3x3_10431373: &
contract_16_3x3_10431375:0
expand_17_5x5_10431380:	 $
expand_17_5x5_10431382: 2
contract_17_3x3_10431385: &
contract_17_3x3_10431387:0
expand_18_5x5_10431392:	 $
expand_18_5x5_10431394: 2
contract_18_3x3_10431397: &
contract_18_3x3_10431399:0
expand_19_5x5_10431404:	 $
expand_19_5x5_10431406: 2
contract_19_3x3_10431409: &
contract_19_3x3_10431411:0
expand_20_5x5_10431416:	 $
expand_20_5x5_10431418: 2
contract_20_3x3_10431421: &
contract_20_3x3_10431423:4
policy_aggregator_10431428:(
policy_aggregator_10431430:-
border_off_10431434:!
border_off_10431436:&
value_head_10431442:	т,!
value_head_10431444:
identity

identity_1ѕб"border_off/StatefulPartitionedCallб'contract_10_3x3/StatefulPartitionedCallб'contract_11_3x3/StatefulPartitionedCallб'contract_12_3x3/StatefulPartitionedCallб'contract_13_3x3/StatefulPartitionedCallб'contract_14_3x3/StatefulPartitionedCallб'contract_15_3x3/StatefulPartitionedCallб'contract_16_3x3/StatefulPartitionedCallб'contract_17_3x3/StatefulPartitionedCallб'contract_18_3x3/StatefulPartitionedCallб'contract_19_3x3/StatefulPartitionedCallб&contract_1_5x5/StatefulPartitionedCallб'contract_20_3x3/StatefulPartitionedCallб&contract_2_3x3/StatefulPartitionedCallб&contract_3_3x3/StatefulPartitionedCallб&contract_4_3x3/StatefulPartitionedCallб&contract_5_3x3/StatefulPartitionedCallб&contract_6_3x3/StatefulPartitionedCallб&contract_7_3x3/StatefulPartitionedCallб&contract_8_3x3/StatefulPartitionedCallб&contract_9_3x3/StatefulPartitionedCallб%expand_10_5x5/StatefulPartitionedCallб%expand_11_5x5/StatefulPartitionedCallб%expand_12_5x5/StatefulPartitionedCallб%expand_13_5x5/StatefulPartitionedCallб%expand_14_5x5/StatefulPartitionedCallб%expand_15_5x5/StatefulPartitionedCallб%expand_16_5x5/StatefulPartitionedCallб%expand_17_5x5/StatefulPartitionedCallб%expand_18_5x5/StatefulPartitionedCallб%expand_19_5x5/StatefulPartitionedCallб&expand_1_11x11/StatefulPartitionedCallб%expand_20_5x5/StatefulPartitionedCallб$expand_2_5x5/StatefulPartitionedCallб$expand_3_5x5/StatefulPartitionedCallб$expand_4_5x5/StatefulPartitionedCallб$expand_5_5x5/StatefulPartitionedCallб$expand_6_5x5/StatefulPartitionedCallб$expand_7_5x5/StatefulPartitionedCallб$expand_8_5x5/StatefulPartitionedCallб$expand_9_5x5/StatefulPartitionedCallб*heuristic_detector/StatefulPartitionedCallб*heuristic_priority/StatefulPartitionedCallб)policy_aggregator/StatefulPartitionedCallб"value_head/StatefulPartitionedCallџ
&expand_1_11x11/StatefulPartitionedCallStatefulPartitionedCallinputsexpand_1_11x11_10431179expand_1_11x11_10431181*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_expand_1_11x11_layer_call_and_return_conditional_losses_10428949ф
*heuristic_detector/StatefulPartitionedCallStatefulPartitionedCallinputsheuristic_detector_10431184heuristic_detector_10431186*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         │*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_heuristic_detector_layer_call_and_return_conditional_losses_10428966о
*heuristic_priority/StatefulPartitionedCallStatefulPartitionedCall3heuristic_detector/StatefulPartitionedCall:output:0heuristic_priority_10431189heuristic_priority_10431191*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_heuristic_priority_layer_call_and_return_conditional_losses_10428983┬
&contract_1_5x5/StatefulPartitionedCallStatefulPartitionedCall/expand_1_11x11/StatefulPartitionedCall:output:0contract_1_5x5_10431194contract_1_5x5_10431196*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_1_5x5_layer_call_and_return_conditional_losses_10429000░
concatenate_19/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0/contract_1_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_19_layer_call_and_return_conditional_losses_10429013▓
$expand_2_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_19/PartitionedCall:output:0expand_2_5x5_10431200expand_2_5x5_10431202*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_2_5x5_layer_call_and_return_conditional_losses_10429026└
&contract_2_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_2_5x5/StatefulPartitionedCall:output:0contract_2_3x3_10431205contract_2_3x3_10431207*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_2_3x3_layer_call_and_return_conditional_losses_10429043ю
skip_2/PartitionedCallPartitionedCall/contract_2_3x3/StatefulPartitionedCall:output:0/contract_1_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_2_layer_call_and_return_conditional_losses_10429055а
concatenate_20/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_20_layer_call_and_return_conditional_losses_10429064▓
$expand_3_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_20/PartitionedCall:output:0expand_3_5x5_10431212expand_3_5x5_10431214*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_3_5x5_layer_call_and_return_conditional_losses_10429077└
&contract_3_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_3_5x5/StatefulPartitionedCall:output:0contract_3_3x3_10431217contract_3_3x3_10431219*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_3_3x3_layer_call_and_return_conditional_losses_10429094ї
skip_3/PartitionedCallPartitionedCall/contract_3_3x3/StatefulPartitionedCall:output:0skip_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_3_layer_call_and_return_conditional_losses_10429106а
concatenate_21/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_21_layer_call_and_return_conditional_losses_10429115▓
$expand_4_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_21/PartitionedCall:output:0expand_4_5x5_10431224expand_4_5x5_10431226*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_4_5x5_layer_call_and_return_conditional_losses_10429128└
&contract_4_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_4_5x5/StatefulPartitionedCall:output:0contract_4_3x3_10431229contract_4_3x3_10431231*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_4_3x3_layer_call_and_return_conditional_losses_10429145ї
skip_4/PartitionedCallPartitionedCall/contract_4_3x3/StatefulPartitionedCall:output:0skip_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_4_layer_call_and_return_conditional_losses_10429157а
concatenate_22/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_22_layer_call_and_return_conditional_losses_10429166▓
$expand_5_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_22/PartitionedCall:output:0expand_5_5x5_10431236expand_5_5x5_10431238*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_5_5x5_layer_call_and_return_conditional_losses_10429179└
&contract_5_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_5_5x5/StatefulPartitionedCall:output:0contract_5_3x3_10431241contract_5_3x3_10431243*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_5_3x3_layer_call_and_return_conditional_losses_10429196ї
skip_5/PartitionedCallPartitionedCall/contract_5_3x3/StatefulPartitionedCall:output:0skip_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_5_layer_call_and_return_conditional_losses_10429208а
concatenate_23/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_23_layer_call_and_return_conditional_losses_10429217▓
$expand_6_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_23/PartitionedCall:output:0expand_6_5x5_10431248expand_6_5x5_10431250*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_6_5x5_layer_call_and_return_conditional_losses_10429230└
&contract_6_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_6_5x5/StatefulPartitionedCall:output:0contract_6_3x3_10431253contract_6_3x3_10431255*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_6_3x3_layer_call_and_return_conditional_losses_10429247ї
skip_6/PartitionedCallPartitionedCall/contract_6_3x3/StatefulPartitionedCall:output:0skip_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_6_layer_call_and_return_conditional_losses_10429259а
concatenate_24/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_24_layer_call_and_return_conditional_losses_10429268▓
$expand_7_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_24/PartitionedCall:output:0expand_7_5x5_10431260expand_7_5x5_10431262*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_7_5x5_layer_call_and_return_conditional_losses_10429281└
&contract_7_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_7_5x5/StatefulPartitionedCall:output:0contract_7_3x3_10431265contract_7_3x3_10431267*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_7_3x3_layer_call_and_return_conditional_losses_10429298ї
skip_7/PartitionedCallPartitionedCall/contract_7_3x3/StatefulPartitionedCall:output:0skip_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_7_layer_call_and_return_conditional_losses_10429310а
concatenate_25/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_25_layer_call_and_return_conditional_losses_10429319▓
$expand_8_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_25/PartitionedCall:output:0expand_8_5x5_10431272expand_8_5x5_10431274*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_8_5x5_layer_call_and_return_conditional_losses_10429332└
&contract_8_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_8_5x5/StatefulPartitionedCall:output:0contract_8_3x3_10431277contract_8_3x3_10431279*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_8_3x3_layer_call_and_return_conditional_losses_10429349ї
skip_8/PartitionedCallPartitionedCall/contract_8_3x3/StatefulPartitionedCall:output:0skip_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_8_layer_call_and_return_conditional_losses_10429361а
concatenate_26/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_26_layer_call_and_return_conditional_losses_10429370▓
$expand_9_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_26/PartitionedCall:output:0expand_9_5x5_10431284expand_9_5x5_10431286*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_9_5x5_layer_call_and_return_conditional_losses_10429383└
&contract_9_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_9_5x5/StatefulPartitionedCall:output:0contract_9_3x3_10431289contract_9_3x3_10431291*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_9_3x3_layer_call_and_return_conditional_losses_10429400ї
skip_9/PartitionedCallPartitionedCall/contract_9_3x3/StatefulPartitionedCall:output:0skip_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_9_layer_call_and_return_conditional_losses_10429412а
concatenate_27/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_27_layer_call_and_return_conditional_losses_10429421Х
%expand_10_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_27/PartitionedCall:output:0expand_10_5x5_10431296expand_10_5x5_10431298*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_10_5x5_layer_call_and_return_conditional_losses_10429434┼
'contract_10_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_10_5x5/StatefulPartitionedCall:output:0contract_10_3x3_10431301contract_10_3x3_10431303*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_10_3x3_layer_call_and_return_conditional_losses_10429451Ј
skip_10/PartitionedCallPartitionedCall0contract_10_3x3/StatefulPartitionedCall:output:0skip_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_10_layer_call_and_return_conditional_losses_10429463А
concatenate_28/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_28_layer_call_and_return_conditional_losses_10429472Х
%expand_11_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_28/PartitionedCall:output:0expand_11_5x5_10431308expand_11_5x5_10431310*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_11_5x5_layer_call_and_return_conditional_losses_10429485┼
'contract_11_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_11_5x5/StatefulPartitionedCall:output:0contract_11_3x3_10431313contract_11_3x3_10431315*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_11_3x3_layer_call_and_return_conditional_losses_10429502љ
skip_11/PartitionedCallPartitionedCall0contract_11_3x3/StatefulPartitionedCall:output:0 skip_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_11_layer_call_and_return_conditional_losses_10429514А
concatenate_29/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_29_layer_call_and_return_conditional_losses_10429523Х
%expand_12_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_29/PartitionedCall:output:0expand_12_5x5_10431320expand_12_5x5_10431322*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_12_5x5_layer_call_and_return_conditional_losses_10429536┼
'contract_12_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_12_5x5/StatefulPartitionedCall:output:0contract_12_3x3_10431325contract_12_3x3_10431327*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_12_3x3_layer_call_and_return_conditional_losses_10429553љ
skip_12/PartitionedCallPartitionedCall0contract_12_3x3/StatefulPartitionedCall:output:0 skip_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_12_layer_call_and_return_conditional_losses_10429565А
concatenate_30/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_30_layer_call_and_return_conditional_losses_10429574Х
%expand_13_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_30/PartitionedCall:output:0expand_13_5x5_10431332expand_13_5x5_10431334*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_13_5x5_layer_call_and_return_conditional_losses_10429587┼
'contract_13_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_13_5x5/StatefulPartitionedCall:output:0contract_13_3x3_10431337contract_13_3x3_10431339*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_13_3x3_layer_call_and_return_conditional_losses_10429604љ
skip_13/PartitionedCallPartitionedCall0contract_13_3x3/StatefulPartitionedCall:output:0 skip_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_13_layer_call_and_return_conditional_losses_10429616А
concatenate_31/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_31_layer_call_and_return_conditional_losses_10429625Х
%expand_14_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_31/PartitionedCall:output:0expand_14_5x5_10431344expand_14_5x5_10431346*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_14_5x5_layer_call_and_return_conditional_losses_10429638┼
'contract_14_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_14_5x5/StatefulPartitionedCall:output:0contract_14_3x3_10431349contract_14_3x3_10431351*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_14_3x3_layer_call_and_return_conditional_losses_10429655љ
skip_14/PartitionedCallPartitionedCall0contract_14_3x3/StatefulPartitionedCall:output:0 skip_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_14_layer_call_and_return_conditional_losses_10429667А
concatenate_32/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_14/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_32_layer_call_and_return_conditional_losses_10429676Х
%expand_15_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_32/PartitionedCall:output:0expand_15_5x5_10431356expand_15_5x5_10431358*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_15_5x5_layer_call_and_return_conditional_losses_10429689┼
'contract_15_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_15_5x5/StatefulPartitionedCall:output:0contract_15_3x3_10431361contract_15_3x3_10431363*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_15_3x3_layer_call_and_return_conditional_losses_10429706љ
skip_15/PartitionedCallPartitionedCall0contract_15_3x3/StatefulPartitionedCall:output:0 skip_14/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_15_layer_call_and_return_conditional_losses_10429718А
concatenate_33/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_33_layer_call_and_return_conditional_losses_10429727Х
%expand_16_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_33/PartitionedCall:output:0expand_16_5x5_10431368expand_16_5x5_10431370*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_16_5x5_layer_call_and_return_conditional_losses_10429740┼
'contract_16_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_16_5x5/StatefulPartitionedCall:output:0contract_16_3x3_10431373contract_16_3x3_10431375*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_16_3x3_layer_call_and_return_conditional_losses_10429757љ
skip_16/PartitionedCallPartitionedCall0contract_16_3x3/StatefulPartitionedCall:output:0 skip_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_16_layer_call_and_return_conditional_losses_10429769А
concatenate_34/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_34_layer_call_and_return_conditional_losses_10429778Х
%expand_17_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_34/PartitionedCall:output:0expand_17_5x5_10431380expand_17_5x5_10431382*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_17_5x5_layer_call_and_return_conditional_losses_10429791┼
'contract_17_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_17_5x5/StatefulPartitionedCall:output:0contract_17_3x3_10431385contract_17_3x3_10431387*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_17_3x3_layer_call_and_return_conditional_losses_10429808љ
skip_17/PartitionedCallPartitionedCall0contract_17_3x3/StatefulPartitionedCall:output:0 skip_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_17_layer_call_and_return_conditional_losses_10429820А
concatenate_35/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_35_layer_call_and_return_conditional_losses_10429829Х
%expand_18_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_35/PartitionedCall:output:0expand_18_5x5_10431392expand_18_5x5_10431394*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_18_5x5_layer_call_and_return_conditional_losses_10429842┼
'contract_18_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_18_5x5/StatefulPartitionedCall:output:0contract_18_3x3_10431397contract_18_3x3_10431399*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_18_3x3_layer_call_and_return_conditional_losses_10429859љ
skip_18/PartitionedCallPartitionedCall0contract_18_3x3/StatefulPartitionedCall:output:0 skip_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_18_layer_call_and_return_conditional_losses_10429871А
concatenate_36/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_18/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_36_layer_call_and_return_conditional_losses_10429880Х
%expand_19_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_36/PartitionedCall:output:0expand_19_5x5_10431404expand_19_5x5_10431406*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_19_5x5_layer_call_and_return_conditional_losses_10429893┼
'contract_19_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_19_5x5/StatefulPartitionedCall:output:0contract_19_3x3_10431409contract_19_3x3_10431411*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_19_3x3_layer_call_and_return_conditional_losses_10429910љ
skip_19/PartitionedCallPartitionedCall0contract_19_3x3/StatefulPartitionedCall:output:0 skip_18/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_19_layer_call_and_return_conditional_losses_10429922А
concatenate_37/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_19/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_37_layer_call_and_return_conditional_losses_10429931Х
%expand_20_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_37/PartitionedCall:output:0expand_20_5x5_10431416expand_20_5x5_10431418*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_20_5x5_layer_call_and_return_conditional_losses_10429944┼
'contract_20_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_20_5x5/StatefulPartitionedCall:output:0contract_20_3x3_10431421contract_20_3x3_10431423*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_20_3x3_layer_call_and_return_conditional_losses_10429961љ
skip_20/PartitionedCallPartitionedCall0contract_20_3x3/StatefulPartitionedCall:output:0 skip_19/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_20_layer_call_and_return_conditional_losses_10429973Д
all_value_input/PartitionedCallPartitionedCall0contract_20_3x3/StatefulPartitionedCall:output:0'concatenate_19/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_all_value_input_layer_call_and_return_conditional_losses_10429982┐
)policy_aggregator/StatefulPartitionedCallStatefulPartitionedCall skip_20/PartitionedCall:output:0policy_aggregator_10431428policy_aggregator_10431430*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_policy_aggregator_layer_call_and_return_conditional_losses_10429995­
 flat_value_input/PartitionedCallPartitionedCall(all_value_input/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         т,* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_flat_value_input_layer_call_and_return_conditional_losses_10430007х
"border_off/StatefulPartitionedCallStatefulPartitionedCall2policy_aggregator/StatefulPartitionedCall:output:0border_off_10431434border_off_10431436*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_border_off_layer_call_and_return_conditional_losses_10430019`
tf.math.truediv_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ╚Bе
tf.math.truediv_1/truedivRealDiv)flat_value_input/PartitionedCall:output:0$tf.math.truediv_1/truediv/y:output:0*
T0*(
_output_shapes
:         т,ж
flat_logits/PartitionedCallPartitionedCall+border_off/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ж* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_flat_logits_layer_call_and_return_conditional_losses_10430033ў
"value_head/StatefulPartitionedCallStatefulPartitionedCalltf.math.truediv_1/truediv:z:0value_head_10431442value_head_10431444*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_value_head_layer_call_and_return_conditional_losses_10430046Р
policy_head/PartitionedCallPartitionedCall$flat_logits/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ж* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_policy_head_layer_call_and_return_conditional_losses_10430057t
IdentityIdentity$policy_head/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ж|

Identity_1Identity+value_head/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Ь
NoOpNoOp#^border_off/StatefulPartitionedCall(^contract_10_3x3/StatefulPartitionedCall(^contract_11_3x3/StatefulPartitionedCall(^contract_12_3x3/StatefulPartitionedCall(^contract_13_3x3/StatefulPartitionedCall(^contract_14_3x3/StatefulPartitionedCall(^contract_15_3x3/StatefulPartitionedCall(^contract_16_3x3/StatefulPartitionedCall(^contract_17_3x3/StatefulPartitionedCall(^contract_18_3x3/StatefulPartitionedCall(^contract_19_3x3/StatefulPartitionedCall'^contract_1_5x5/StatefulPartitionedCall(^contract_20_3x3/StatefulPartitionedCall'^contract_2_3x3/StatefulPartitionedCall'^contract_3_3x3/StatefulPartitionedCall'^contract_4_3x3/StatefulPartitionedCall'^contract_5_3x3/StatefulPartitionedCall'^contract_6_3x3/StatefulPartitionedCall'^contract_7_3x3/StatefulPartitionedCall'^contract_8_3x3/StatefulPartitionedCall'^contract_9_3x3/StatefulPartitionedCall&^expand_10_5x5/StatefulPartitionedCall&^expand_11_5x5/StatefulPartitionedCall&^expand_12_5x5/StatefulPartitionedCall&^expand_13_5x5/StatefulPartitionedCall&^expand_14_5x5/StatefulPartitionedCall&^expand_15_5x5/StatefulPartitionedCall&^expand_16_5x5/StatefulPartitionedCall&^expand_17_5x5/StatefulPartitionedCall&^expand_18_5x5/StatefulPartitionedCall&^expand_19_5x5/StatefulPartitionedCall'^expand_1_11x11/StatefulPartitionedCall&^expand_20_5x5/StatefulPartitionedCall%^expand_2_5x5/StatefulPartitionedCall%^expand_3_5x5/StatefulPartitionedCall%^expand_4_5x5/StatefulPartitionedCall%^expand_5_5x5/StatefulPartitionedCall%^expand_6_5x5/StatefulPartitionedCall%^expand_7_5x5/StatefulPartitionedCall%^expand_8_5x5/StatefulPartitionedCall%^expand_9_5x5/StatefulPartitionedCall+^heuristic_detector/StatefulPartitionedCall+^heuristic_priority/StatefulPartitionedCall*^policy_aggregator/StatefulPartitionedCall#^value_head/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*С
_input_shapesм
¤:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"border_off/StatefulPartitionedCall"border_off/StatefulPartitionedCall2R
'contract_10_3x3/StatefulPartitionedCall'contract_10_3x3/StatefulPartitionedCall2R
'contract_11_3x3/StatefulPartitionedCall'contract_11_3x3/StatefulPartitionedCall2R
'contract_12_3x3/StatefulPartitionedCall'contract_12_3x3/StatefulPartitionedCall2R
'contract_13_3x3/StatefulPartitionedCall'contract_13_3x3/StatefulPartitionedCall2R
'contract_14_3x3/StatefulPartitionedCall'contract_14_3x3/StatefulPartitionedCall2R
'contract_15_3x3/StatefulPartitionedCall'contract_15_3x3/StatefulPartitionedCall2R
'contract_16_3x3/StatefulPartitionedCall'contract_16_3x3/StatefulPartitionedCall2R
'contract_17_3x3/StatefulPartitionedCall'contract_17_3x3/StatefulPartitionedCall2R
'contract_18_3x3/StatefulPartitionedCall'contract_18_3x3/StatefulPartitionedCall2R
'contract_19_3x3/StatefulPartitionedCall'contract_19_3x3/StatefulPartitionedCall2P
&contract_1_5x5/StatefulPartitionedCall&contract_1_5x5/StatefulPartitionedCall2R
'contract_20_3x3/StatefulPartitionedCall'contract_20_3x3/StatefulPartitionedCall2P
&contract_2_3x3/StatefulPartitionedCall&contract_2_3x3/StatefulPartitionedCall2P
&contract_3_3x3/StatefulPartitionedCall&contract_3_3x3/StatefulPartitionedCall2P
&contract_4_3x3/StatefulPartitionedCall&contract_4_3x3/StatefulPartitionedCall2P
&contract_5_3x3/StatefulPartitionedCall&contract_5_3x3/StatefulPartitionedCall2P
&contract_6_3x3/StatefulPartitionedCall&contract_6_3x3/StatefulPartitionedCall2P
&contract_7_3x3/StatefulPartitionedCall&contract_7_3x3/StatefulPartitionedCall2P
&contract_8_3x3/StatefulPartitionedCall&contract_8_3x3/StatefulPartitionedCall2P
&contract_9_3x3/StatefulPartitionedCall&contract_9_3x3/StatefulPartitionedCall2N
%expand_10_5x5/StatefulPartitionedCall%expand_10_5x5/StatefulPartitionedCall2N
%expand_11_5x5/StatefulPartitionedCall%expand_11_5x5/StatefulPartitionedCall2N
%expand_12_5x5/StatefulPartitionedCall%expand_12_5x5/StatefulPartitionedCall2N
%expand_13_5x5/StatefulPartitionedCall%expand_13_5x5/StatefulPartitionedCall2N
%expand_14_5x5/StatefulPartitionedCall%expand_14_5x5/StatefulPartitionedCall2N
%expand_15_5x5/StatefulPartitionedCall%expand_15_5x5/StatefulPartitionedCall2N
%expand_16_5x5/StatefulPartitionedCall%expand_16_5x5/StatefulPartitionedCall2N
%expand_17_5x5/StatefulPartitionedCall%expand_17_5x5/StatefulPartitionedCall2N
%expand_18_5x5/StatefulPartitionedCall%expand_18_5x5/StatefulPartitionedCall2N
%expand_19_5x5/StatefulPartitionedCall%expand_19_5x5/StatefulPartitionedCall2P
&expand_1_11x11/StatefulPartitionedCall&expand_1_11x11/StatefulPartitionedCall2N
%expand_20_5x5/StatefulPartitionedCall%expand_20_5x5/StatefulPartitionedCall2L
$expand_2_5x5/StatefulPartitionedCall$expand_2_5x5/StatefulPartitionedCall2L
$expand_3_5x5/StatefulPartitionedCall$expand_3_5x5/StatefulPartitionedCall2L
$expand_4_5x5/StatefulPartitionedCall$expand_4_5x5/StatefulPartitionedCall2L
$expand_5_5x5/StatefulPartitionedCall$expand_5_5x5/StatefulPartitionedCall2L
$expand_6_5x5/StatefulPartitionedCall$expand_6_5x5/StatefulPartitionedCall2L
$expand_7_5x5/StatefulPartitionedCall$expand_7_5x5/StatefulPartitionedCall2L
$expand_8_5x5/StatefulPartitionedCall$expand_8_5x5/StatefulPartitionedCall2L
$expand_9_5x5/StatefulPartitionedCall$expand_9_5x5/StatefulPartitionedCall2X
*heuristic_detector/StatefulPartitionedCall*heuristic_detector/StatefulPartitionedCall2X
*heuristic_priority/StatefulPartitionedCall*heuristic_priority/StatefulPartitionedCall2V
)policy_aggregator/StatefulPartitionedCall)policy_aggregator/StatefulPartitionedCall2H
"value_head/StatefulPartitionedCall"value_head/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
г

Ђ
H__inference_border_off_layer_call_and_return_conditional_losses_10433926

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Ѓ
Е
4__inference_policy_aggregator_layer_call_fn_10433883

inputs!
unknown:
	unknown_0:
identityѕбStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_policy_aggregator_layer_call_and_return_conditional_losses_10429995w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ь
v
L__inference_concatenate_20_layer_call_and_return_conditional_losses_10429064

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :}
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
Ќ
Ё
L__inference_contract_7_3x3_layer_call_and_return_conditional_losses_10429298

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Їё
█.
M__inference_gomoku_resnet_1_layer_call_and_return_conditional_losses_10432096

inputs2
expand_1_11x11_10431825:ђ&
expand_1_11x11_10431827:	ђ6
heuristic_detector_10431830:│*
heuristic_detector_10431832:	│6
heuristic_priority_10431835:│)
heuristic_priority_10431837:2
contract_1_5x5_10431840:ђ%
contract_1_5x5_10431842:/
expand_2_5x5_10431846:	 #
expand_2_5x5_10431848: 1
contract_2_3x3_10431851: %
contract_2_3x3_10431853:/
expand_3_5x5_10431858:	 #
expand_3_5x5_10431860: 1
contract_3_3x3_10431863: %
contract_3_3x3_10431865:/
expand_4_5x5_10431870:	 #
expand_4_5x5_10431872: 1
contract_4_3x3_10431875: %
contract_4_3x3_10431877:/
expand_5_5x5_10431882:	 #
expand_5_5x5_10431884: 1
contract_5_3x3_10431887: %
contract_5_3x3_10431889:/
expand_6_5x5_10431894:	 #
expand_6_5x5_10431896: 1
contract_6_3x3_10431899: %
contract_6_3x3_10431901:/
expand_7_5x5_10431906:	 #
expand_7_5x5_10431908: 1
contract_7_3x3_10431911: %
contract_7_3x3_10431913:/
expand_8_5x5_10431918:	 #
expand_8_5x5_10431920: 1
contract_8_3x3_10431923: %
contract_8_3x3_10431925:/
expand_9_5x5_10431930:	 #
expand_9_5x5_10431932: 1
contract_9_3x3_10431935: %
contract_9_3x3_10431937:0
expand_10_5x5_10431942:	 $
expand_10_5x5_10431944: 2
contract_10_3x3_10431947: &
contract_10_3x3_10431949:0
expand_11_5x5_10431954:	 $
expand_11_5x5_10431956: 2
contract_11_3x3_10431959: &
contract_11_3x3_10431961:0
expand_12_5x5_10431966:	 $
expand_12_5x5_10431968: 2
contract_12_3x3_10431971: &
contract_12_3x3_10431973:0
expand_13_5x5_10431978:	 $
expand_13_5x5_10431980: 2
contract_13_3x3_10431983: &
contract_13_3x3_10431985:0
expand_14_5x5_10431990:	 $
expand_14_5x5_10431992: 2
contract_14_3x3_10431995: &
contract_14_3x3_10431997:0
expand_15_5x5_10432002:	 $
expand_15_5x5_10432004: 2
contract_15_3x3_10432007: &
contract_15_3x3_10432009:0
expand_16_5x5_10432014:	 $
expand_16_5x5_10432016: 2
contract_16_3x3_10432019: &
contract_16_3x3_10432021:0
expand_17_5x5_10432026:	 $
expand_17_5x5_10432028: 2
contract_17_3x3_10432031: &
contract_17_3x3_10432033:0
expand_18_5x5_10432038:	 $
expand_18_5x5_10432040: 2
contract_18_3x3_10432043: &
contract_18_3x3_10432045:0
expand_19_5x5_10432050:	 $
expand_19_5x5_10432052: 2
contract_19_3x3_10432055: &
contract_19_3x3_10432057:0
expand_20_5x5_10432062:	 $
expand_20_5x5_10432064: 2
contract_20_3x3_10432067: &
contract_20_3x3_10432069:4
policy_aggregator_10432074:(
policy_aggregator_10432076:-
border_off_10432080:!
border_off_10432082:&
value_head_10432088:	т,!
value_head_10432090:
identity

identity_1ѕб"border_off/StatefulPartitionedCallб'contract_10_3x3/StatefulPartitionedCallб'contract_11_3x3/StatefulPartitionedCallб'contract_12_3x3/StatefulPartitionedCallб'contract_13_3x3/StatefulPartitionedCallб'contract_14_3x3/StatefulPartitionedCallб'contract_15_3x3/StatefulPartitionedCallб'contract_16_3x3/StatefulPartitionedCallб'contract_17_3x3/StatefulPartitionedCallб'contract_18_3x3/StatefulPartitionedCallб'contract_19_3x3/StatefulPartitionedCallб&contract_1_5x5/StatefulPartitionedCallб'contract_20_3x3/StatefulPartitionedCallб&contract_2_3x3/StatefulPartitionedCallб&contract_3_3x3/StatefulPartitionedCallб&contract_4_3x3/StatefulPartitionedCallб&contract_5_3x3/StatefulPartitionedCallб&contract_6_3x3/StatefulPartitionedCallб&contract_7_3x3/StatefulPartitionedCallб&contract_8_3x3/StatefulPartitionedCallб&contract_9_3x3/StatefulPartitionedCallб%expand_10_5x5/StatefulPartitionedCallб%expand_11_5x5/StatefulPartitionedCallб%expand_12_5x5/StatefulPartitionedCallб%expand_13_5x5/StatefulPartitionedCallб%expand_14_5x5/StatefulPartitionedCallб%expand_15_5x5/StatefulPartitionedCallб%expand_16_5x5/StatefulPartitionedCallб%expand_17_5x5/StatefulPartitionedCallб%expand_18_5x5/StatefulPartitionedCallб%expand_19_5x5/StatefulPartitionedCallб&expand_1_11x11/StatefulPartitionedCallб%expand_20_5x5/StatefulPartitionedCallб$expand_2_5x5/StatefulPartitionedCallб$expand_3_5x5/StatefulPartitionedCallб$expand_4_5x5/StatefulPartitionedCallб$expand_5_5x5/StatefulPartitionedCallб$expand_6_5x5/StatefulPartitionedCallб$expand_7_5x5/StatefulPartitionedCallб$expand_8_5x5/StatefulPartitionedCallб$expand_9_5x5/StatefulPartitionedCallб*heuristic_detector/StatefulPartitionedCallб*heuristic_priority/StatefulPartitionedCallб)policy_aggregator/StatefulPartitionedCallб"value_head/StatefulPartitionedCallџ
&expand_1_11x11/StatefulPartitionedCallStatefulPartitionedCallinputsexpand_1_11x11_10431825expand_1_11x11_10431827*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_expand_1_11x11_layer_call_and_return_conditional_losses_10428949ф
*heuristic_detector/StatefulPartitionedCallStatefulPartitionedCallinputsheuristic_detector_10431830heuristic_detector_10431832*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         │*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_heuristic_detector_layer_call_and_return_conditional_losses_10428966о
*heuristic_priority/StatefulPartitionedCallStatefulPartitionedCall3heuristic_detector/StatefulPartitionedCall:output:0heuristic_priority_10431835heuristic_priority_10431837*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_heuristic_priority_layer_call_and_return_conditional_losses_10428983┬
&contract_1_5x5/StatefulPartitionedCallStatefulPartitionedCall/expand_1_11x11/StatefulPartitionedCall:output:0contract_1_5x5_10431840contract_1_5x5_10431842*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_1_5x5_layer_call_and_return_conditional_losses_10429000░
concatenate_19/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0/contract_1_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_19_layer_call_and_return_conditional_losses_10429013▓
$expand_2_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_19/PartitionedCall:output:0expand_2_5x5_10431846expand_2_5x5_10431848*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_2_5x5_layer_call_and_return_conditional_losses_10429026└
&contract_2_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_2_5x5/StatefulPartitionedCall:output:0contract_2_3x3_10431851contract_2_3x3_10431853*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_2_3x3_layer_call_and_return_conditional_losses_10429043ю
skip_2/PartitionedCallPartitionedCall/contract_2_3x3/StatefulPartitionedCall:output:0/contract_1_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_2_layer_call_and_return_conditional_losses_10429055а
concatenate_20/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_20_layer_call_and_return_conditional_losses_10429064▓
$expand_3_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_20/PartitionedCall:output:0expand_3_5x5_10431858expand_3_5x5_10431860*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_3_5x5_layer_call_and_return_conditional_losses_10429077└
&contract_3_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_3_5x5/StatefulPartitionedCall:output:0contract_3_3x3_10431863contract_3_3x3_10431865*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_3_3x3_layer_call_and_return_conditional_losses_10429094ї
skip_3/PartitionedCallPartitionedCall/contract_3_3x3/StatefulPartitionedCall:output:0skip_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_3_layer_call_and_return_conditional_losses_10429106а
concatenate_21/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_21_layer_call_and_return_conditional_losses_10429115▓
$expand_4_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_21/PartitionedCall:output:0expand_4_5x5_10431870expand_4_5x5_10431872*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_4_5x5_layer_call_and_return_conditional_losses_10429128└
&contract_4_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_4_5x5/StatefulPartitionedCall:output:0contract_4_3x3_10431875contract_4_3x3_10431877*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_4_3x3_layer_call_and_return_conditional_losses_10429145ї
skip_4/PartitionedCallPartitionedCall/contract_4_3x3/StatefulPartitionedCall:output:0skip_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_4_layer_call_and_return_conditional_losses_10429157а
concatenate_22/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_22_layer_call_and_return_conditional_losses_10429166▓
$expand_5_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_22/PartitionedCall:output:0expand_5_5x5_10431882expand_5_5x5_10431884*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_5_5x5_layer_call_and_return_conditional_losses_10429179└
&contract_5_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_5_5x5/StatefulPartitionedCall:output:0contract_5_3x3_10431887contract_5_3x3_10431889*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_5_3x3_layer_call_and_return_conditional_losses_10429196ї
skip_5/PartitionedCallPartitionedCall/contract_5_3x3/StatefulPartitionedCall:output:0skip_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_5_layer_call_and_return_conditional_losses_10429208а
concatenate_23/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_23_layer_call_and_return_conditional_losses_10429217▓
$expand_6_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_23/PartitionedCall:output:0expand_6_5x5_10431894expand_6_5x5_10431896*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_6_5x5_layer_call_and_return_conditional_losses_10429230└
&contract_6_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_6_5x5/StatefulPartitionedCall:output:0contract_6_3x3_10431899contract_6_3x3_10431901*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_6_3x3_layer_call_and_return_conditional_losses_10429247ї
skip_6/PartitionedCallPartitionedCall/contract_6_3x3/StatefulPartitionedCall:output:0skip_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_6_layer_call_and_return_conditional_losses_10429259а
concatenate_24/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_24_layer_call_and_return_conditional_losses_10429268▓
$expand_7_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_24/PartitionedCall:output:0expand_7_5x5_10431906expand_7_5x5_10431908*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_7_5x5_layer_call_and_return_conditional_losses_10429281└
&contract_7_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_7_5x5/StatefulPartitionedCall:output:0contract_7_3x3_10431911contract_7_3x3_10431913*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_7_3x3_layer_call_and_return_conditional_losses_10429298ї
skip_7/PartitionedCallPartitionedCall/contract_7_3x3/StatefulPartitionedCall:output:0skip_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_7_layer_call_and_return_conditional_losses_10429310а
concatenate_25/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_25_layer_call_and_return_conditional_losses_10429319▓
$expand_8_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_25/PartitionedCall:output:0expand_8_5x5_10431918expand_8_5x5_10431920*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_8_5x5_layer_call_and_return_conditional_losses_10429332└
&contract_8_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_8_5x5/StatefulPartitionedCall:output:0contract_8_3x3_10431923contract_8_3x3_10431925*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_8_3x3_layer_call_and_return_conditional_losses_10429349ї
skip_8/PartitionedCallPartitionedCall/contract_8_3x3/StatefulPartitionedCall:output:0skip_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_8_layer_call_and_return_conditional_losses_10429361а
concatenate_26/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_26_layer_call_and_return_conditional_losses_10429370▓
$expand_9_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_26/PartitionedCall:output:0expand_9_5x5_10431930expand_9_5x5_10431932*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_9_5x5_layer_call_and_return_conditional_losses_10429383└
&contract_9_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_9_5x5/StatefulPartitionedCall:output:0contract_9_3x3_10431935contract_9_3x3_10431937*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_9_3x3_layer_call_and_return_conditional_losses_10429400ї
skip_9/PartitionedCallPartitionedCall/contract_9_3x3/StatefulPartitionedCall:output:0skip_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_9_layer_call_and_return_conditional_losses_10429412а
concatenate_27/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_27_layer_call_and_return_conditional_losses_10429421Х
%expand_10_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_27/PartitionedCall:output:0expand_10_5x5_10431942expand_10_5x5_10431944*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_10_5x5_layer_call_and_return_conditional_losses_10429434┼
'contract_10_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_10_5x5/StatefulPartitionedCall:output:0contract_10_3x3_10431947contract_10_3x3_10431949*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_10_3x3_layer_call_and_return_conditional_losses_10429451Ј
skip_10/PartitionedCallPartitionedCall0contract_10_3x3/StatefulPartitionedCall:output:0skip_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_10_layer_call_and_return_conditional_losses_10429463А
concatenate_28/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_28_layer_call_and_return_conditional_losses_10429472Х
%expand_11_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_28/PartitionedCall:output:0expand_11_5x5_10431954expand_11_5x5_10431956*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_11_5x5_layer_call_and_return_conditional_losses_10429485┼
'contract_11_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_11_5x5/StatefulPartitionedCall:output:0contract_11_3x3_10431959contract_11_3x3_10431961*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_11_3x3_layer_call_and_return_conditional_losses_10429502љ
skip_11/PartitionedCallPartitionedCall0contract_11_3x3/StatefulPartitionedCall:output:0 skip_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_11_layer_call_and_return_conditional_losses_10429514А
concatenate_29/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_29_layer_call_and_return_conditional_losses_10429523Х
%expand_12_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_29/PartitionedCall:output:0expand_12_5x5_10431966expand_12_5x5_10431968*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_12_5x5_layer_call_and_return_conditional_losses_10429536┼
'contract_12_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_12_5x5/StatefulPartitionedCall:output:0contract_12_3x3_10431971contract_12_3x3_10431973*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_12_3x3_layer_call_and_return_conditional_losses_10429553љ
skip_12/PartitionedCallPartitionedCall0contract_12_3x3/StatefulPartitionedCall:output:0 skip_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_12_layer_call_and_return_conditional_losses_10429565А
concatenate_30/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_30_layer_call_and_return_conditional_losses_10429574Х
%expand_13_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_30/PartitionedCall:output:0expand_13_5x5_10431978expand_13_5x5_10431980*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_13_5x5_layer_call_and_return_conditional_losses_10429587┼
'contract_13_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_13_5x5/StatefulPartitionedCall:output:0contract_13_3x3_10431983contract_13_3x3_10431985*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_13_3x3_layer_call_and_return_conditional_losses_10429604љ
skip_13/PartitionedCallPartitionedCall0contract_13_3x3/StatefulPartitionedCall:output:0 skip_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_13_layer_call_and_return_conditional_losses_10429616А
concatenate_31/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_31_layer_call_and_return_conditional_losses_10429625Х
%expand_14_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_31/PartitionedCall:output:0expand_14_5x5_10431990expand_14_5x5_10431992*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_14_5x5_layer_call_and_return_conditional_losses_10429638┼
'contract_14_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_14_5x5/StatefulPartitionedCall:output:0contract_14_3x3_10431995contract_14_3x3_10431997*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_14_3x3_layer_call_and_return_conditional_losses_10429655љ
skip_14/PartitionedCallPartitionedCall0contract_14_3x3/StatefulPartitionedCall:output:0 skip_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_14_layer_call_and_return_conditional_losses_10429667А
concatenate_32/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_14/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_32_layer_call_and_return_conditional_losses_10429676Х
%expand_15_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_32/PartitionedCall:output:0expand_15_5x5_10432002expand_15_5x5_10432004*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_15_5x5_layer_call_and_return_conditional_losses_10429689┼
'contract_15_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_15_5x5/StatefulPartitionedCall:output:0contract_15_3x3_10432007contract_15_3x3_10432009*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_15_3x3_layer_call_and_return_conditional_losses_10429706љ
skip_15/PartitionedCallPartitionedCall0contract_15_3x3/StatefulPartitionedCall:output:0 skip_14/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_15_layer_call_and_return_conditional_losses_10429718А
concatenate_33/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_33_layer_call_and_return_conditional_losses_10429727Х
%expand_16_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_33/PartitionedCall:output:0expand_16_5x5_10432014expand_16_5x5_10432016*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_16_5x5_layer_call_and_return_conditional_losses_10429740┼
'contract_16_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_16_5x5/StatefulPartitionedCall:output:0contract_16_3x3_10432019contract_16_3x3_10432021*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_16_3x3_layer_call_and_return_conditional_losses_10429757љ
skip_16/PartitionedCallPartitionedCall0contract_16_3x3/StatefulPartitionedCall:output:0 skip_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_16_layer_call_and_return_conditional_losses_10429769А
concatenate_34/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_34_layer_call_and_return_conditional_losses_10429778Х
%expand_17_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_34/PartitionedCall:output:0expand_17_5x5_10432026expand_17_5x5_10432028*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_17_5x5_layer_call_and_return_conditional_losses_10429791┼
'contract_17_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_17_5x5/StatefulPartitionedCall:output:0contract_17_3x3_10432031contract_17_3x3_10432033*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_17_3x3_layer_call_and_return_conditional_losses_10429808љ
skip_17/PartitionedCallPartitionedCall0contract_17_3x3/StatefulPartitionedCall:output:0 skip_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_17_layer_call_and_return_conditional_losses_10429820А
concatenate_35/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_35_layer_call_and_return_conditional_losses_10429829Х
%expand_18_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_35/PartitionedCall:output:0expand_18_5x5_10432038expand_18_5x5_10432040*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_18_5x5_layer_call_and_return_conditional_losses_10429842┼
'contract_18_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_18_5x5/StatefulPartitionedCall:output:0contract_18_3x3_10432043contract_18_3x3_10432045*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_18_3x3_layer_call_and_return_conditional_losses_10429859љ
skip_18/PartitionedCallPartitionedCall0contract_18_3x3/StatefulPartitionedCall:output:0 skip_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_18_layer_call_and_return_conditional_losses_10429871А
concatenate_36/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_18/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_36_layer_call_and_return_conditional_losses_10429880Х
%expand_19_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_36/PartitionedCall:output:0expand_19_5x5_10432050expand_19_5x5_10432052*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_19_5x5_layer_call_and_return_conditional_losses_10429893┼
'contract_19_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_19_5x5/StatefulPartitionedCall:output:0contract_19_3x3_10432055contract_19_3x3_10432057*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_19_3x3_layer_call_and_return_conditional_losses_10429910љ
skip_19/PartitionedCallPartitionedCall0contract_19_3x3/StatefulPartitionedCall:output:0 skip_18/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_19_layer_call_and_return_conditional_losses_10429922А
concatenate_37/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_19/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_37_layer_call_and_return_conditional_losses_10429931Х
%expand_20_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_37/PartitionedCall:output:0expand_20_5x5_10432062expand_20_5x5_10432064*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_20_5x5_layer_call_and_return_conditional_losses_10429944┼
'contract_20_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_20_5x5/StatefulPartitionedCall:output:0contract_20_3x3_10432067contract_20_3x3_10432069*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_20_3x3_layer_call_and_return_conditional_losses_10429961љ
skip_20/PartitionedCallPartitionedCall0contract_20_3x3/StatefulPartitionedCall:output:0 skip_19/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_20_layer_call_and_return_conditional_losses_10429973Д
all_value_input/PartitionedCallPartitionedCall0contract_20_3x3/StatefulPartitionedCall:output:0'concatenate_19/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_all_value_input_layer_call_and_return_conditional_losses_10429982┐
)policy_aggregator/StatefulPartitionedCallStatefulPartitionedCall skip_20/PartitionedCall:output:0policy_aggregator_10432074policy_aggregator_10432076*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_policy_aggregator_layer_call_and_return_conditional_losses_10429995­
 flat_value_input/PartitionedCallPartitionedCall(all_value_input/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         т,* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_flat_value_input_layer_call_and_return_conditional_losses_10430007х
"border_off/StatefulPartitionedCallStatefulPartitionedCall2policy_aggregator/StatefulPartitionedCall:output:0border_off_10432080border_off_10432082*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_border_off_layer_call_and_return_conditional_losses_10430019`
tf.math.truediv_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ╚Bе
tf.math.truediv_1/truedivRealDiv)flat_value_input/PartitionedCall:output:0$tf.math.truediv_1/truediv/y:output:0*
T0*(
_output_shapes
:         т,ж
flat_logits/PartitionedCallPartitionedCall+border_off/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ж* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_flat_logits_layer_call_and_return_conditional_losses_10430033ў
"value_head/StatefulPartitionedCallStatefulPartitionedCalltf.math.truediv_1/truediv:z:0value_head_10432088value_head_10432090*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_value_head_layer_call_and_return_conditional_losses_10430046Р
policy_head/PartitionedCallPartitionedCall$flat_logits/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ж* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_policy_head_layer_call_and_return_conditional_losses_10430057t
IdentityIdentity$policy_head/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ж|

Identity_1Identity+value_head/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Ь
NoOpNoOp#^border_off/StatefulPartitionedCall(^contract_10_3x3/StatefulPartitionedCall(^contract_11_3x3/StatefulPartitionedCall(^contract_12_3x3/StatefulPartitionedCall(^contract_13_3x3/StatefulPartitionedCall(^contract_14_3x3/StatefulPartitionedCall(^contract_15_3x3/StatefulPartitionedCall(^contract_16_3x3/StatefulPartitionedCall(^contract_17_3x3/StatefulPartitionedCall(^contract_18_3x3/StatefulPartitionedCall(^contract_19_3x3/StatefulPartitionedCall'^contract_1_5x5/StatefulPartitionedCall(^contract_20_3x3/StatefulPartitionedCall'^contract_2_3x3/StatefulPartitionedCall'^contract_3_3x3/StatefulPartitionedCall'^contract_4_3x3/StatefulPartitionedCall'^contract_5_3x3/StatefulPartitionedCall'^contract_6_3x3/StatefulPartitionedCall'^contract_7_3x3/StatefulPartitionedCall'^contract_8_3x3/StatefulPartitionedCall'^contract_9_3x3/StatefulPartitionedCall&^expand_10_5x5/StatefulPartitionedCall&^expand_11_5x5/StatefulPartitionedCall&^expand_12_5x5/StatefulPartitionedCall&^expand_13_5x5/StatefulPartitionedCall&^expand_14_5x5/StatefulPartitionedCall&^expand_15_5x5/StatefulPartitionedCall&^expand_16_5x5/StatefulPartitionedCall&^expand_17_5x5/StatefulPartitionedCall&^expand_18_5x5/StatefulPartitionedCall&^expand_19_5x5/StatefulPartitionedCall'^expand_1_11x11/StatefulPartitionedCall&^expand_20_5x5/StatefulPartitionedCall%^expand_2_5x5/StatefulPartitionedCall%^expand_3_5x5/StatefulPartitionedCall%^expand_4_5x5/StatefulPartitionedCall%^expand_5_5x5/StatefulPartitionedCall%^expand_6_5x5/StatefulPartitionedCall%^expand_7_5x5/StatefulPartitionedCall%^expand_8_5x5/StatefulPartitionedCall%^expand_9_5x5/StatefulPartitionedCall+^heuristic_detector/StatefulPartitionedCall+^heuristic_priority/StatefulPartitionedCall*^policy_aggregator/StatefulPartitionedCall#^value_head/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*С
_input_shapesм
¤:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"border_off/StatefulPartitionedCall"border_off/StatefulPartitionedCall2R
'contract_10_3x3/StatefulPartitionedCall'contract_10_3x3/StatefulPartitionedCall2R
'contract_11_3x3/StatefulPartitionedCall'contract_11_3x3/StatefulPartitionedCall2R
'contract_12_3x3/StatefulPartitionedCall'contract_12_3x3/StatefulPartitionedCall2R
'contract_13_3x3/StatefulPartitionedCall'contract_13_3x3/StatefulPartitionedCall2R
'contract_14_3x3/StatefulPartitionedCall'contract_14_3x3/StatefulPartitionedCall2R
'contract_15_3x3/StatefulPartitionedCall'contract_15_3x3/StatefulPartitionedCall2R
'contract_16_3x3/StatefulPartitionedCall'contract_16_3x3/StatefulPartitionedCall2R
'contract_17_3x3/StatefulPartitionedCall'contract_17_3x3/StatefulPartitionedCall2R
'contract_18_3x3/StatefulPartitionedCall'contract_18_3x3/StatefulPartitionedCall2R
'contract_19_3x3/StatefulPartitionedCall'contract_19_3x3/StatefulPartitionedCall2P
&contract_1_5x5/StatefulPartitionedCall&contract_1_5x5/StatefulPartitionedCall2R
'contract_20_3x3/StatefulPartitionedCall'contract_20_3x3/StatefulPartitionedCall2P
&contract_2_3x3/StatefulPartitionedCall&contract_2_3x3/StatefulPartitionedCall2P
&contract_3_3x3/StatefulPartitionedCall&contract_3_3x3/StatefulPartitionedCall2P
&contract_4_3x3/StatefulPartitionedCall&contract_4_3x3/StatefulPartitionedCall2P
&contract_5_3x3/StatefulPartitionedCall&contract_5_3x3/StatefulPartitionedCall2P
&contract_6_3x3/StatefulPartitionedCall&contract_6_3x3/StatefulPartitionedCall2P
&contract_7_3x3/StatefulPartitionedCall&contract_7_3x3/StatefulPartitionedCall2P
&contract_8_3x3/StatefulPartitionedCall&contract_8_3x3/StatefulPartitionedCall2P
&contract_9_3x3/StatefulPartitionedCall&contract_9_3x3/StatefulPartitionedCall2N
%expand_10_5x5/StatefulPartitionedCall%expand_10_5x5/StatefulPartitionedCall2N
%expand_11_5x5/StatefulPartitionedCall%expand_11_5x5/StatefulPartitionedCall2N
%expand_12_5x5/StatefulPartitionedCall%expand_12_5x5/StatefulPartitionedCall2N
%expand_13_5x5/StatefulPartitionedCall%expand_13_5x5/StatefulPartitionedCall2N
%expand_14_5x5/StatefulPartitionedCall%expand_14_5x5/StatefulPartitionedCall2N
%expand_15_5x5/StatefulPartitionedCall%expand_15_5x5/StatefulPartitionedCall2N
%expand_16_5x5/StatefulPartitionedCall%expand_16_5x5/StatefulPartitionedCall2N
%expand_17_5x5/StatefulPartitionedCall%expand_17_5x5/StatefulPartitionedCall2N
%expand_18_5x5/StatefulPartitionedCall%expand_18_5x5/StatefulPartitionedCall2N
%expand_19_5x5/StatefulPartitionedCall%expand_19_5x5/StatefulPartitionedCall2P
&expand_1_11x11/StatefulPartitionedCall&expand_1_11x11/StatefulPartitionedCall2N
%expand_20_5x5/StatefulPartitionedCall%expand_20_5x5/StatefulPartitionedCall2L
$expand_2_5x5/StatefulPartitionedCall$expand_2_5x5/StatefulPartitionedCall2L
$expand_3_5x5/StatefulPartitionedCall$expand_3_5x5/StatefulPartitionedCall2L
$expand_4_5x5/StatefulPartitionedCall$expand_4_5x5/StatefulPartitionedCall2L
$expand_5_5x5/StatefulPartitionedCall$expand_5_5x5/StatefulPartitionedCall2L
$expand_6_5x5/StatefulPartitionedCall$expand_6_5x5/StatefulPartitionedCall2L
$expand_7_5x5/StatefulPartitionedCall$expand_7_5x5/StatefulPartitionedCall2L
$expand_8_5x5/StatefulPartitionedCall$expand_8_5x5/StatefulPartitionedCall2L
$expand_9_5x5/StatefulPartitionedCall$expand_9_5x5/StatefulPartitionedCall2X
*heuristic_detector/StatefulPartitionedCall*heuristic_detector/StatefulPartitionedCall2X
*heuristic_priority/StatefulPartitionedCall*heuristic_priority/StatefulPartitionedCall2V
)policy_aggregator/StatefulPartitionedCall)policy_aggregator/StatefulPartitionedCall2H
"value_head/StatefulPartitionedCall"value_head/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Џ*
Г
2__inference_gomoku_resnet_1_layer_call_fn_10430246

inputs"
unknown:ђ
	unknown_0:	ђ$
	unknown_1:│
	unknown_2:	│$
	unknown_3:│
	unknown_4:$
	unknown_5:ђ
	unknown_6:#
	unknown_7:	 
	unknown_8: #
	unknown_9: 

unknown_10:$

unknown_11:	 

unknown_12: $

unknown_13: 

unknown_14:$

unknown_15:	 

unknown_16: $

unknown_17: 

unknown_18:$

unknown_19:	 

unknown_20: $

unknown_21: 

unknown_22:$

unknown_23:	 

unknown_24: $

unknown_25: 

unknown_26:$

unknown_27:	 

unknown_28: $

unknown_29: 

unknown_30:$

unknown_31:	 

unknown_32: $

unknown_33: 

unknown_34:$

unknown_35:	 

unknown_36: $

unknown_37: 

unknown_38:$

unknown_39:	 

unknown_40: $

unknown_41: 

unknown_42:$

unknown_43:	 

unknown_44: $

unknown_45: 

unknown_46:$

unknown_47:	 

unknown_48: $

unknown_49: 

unknown_50:$

unknown_51:	 

unknown_52: $

unknown_53: 

unknown_54:$

unknown_55:	 

unknown_56: $

unknown_57: 

unknown_58:$

unknown_59:	 

unknown_60: $

unknown_61: 

unknown_62:$

unknown_63:	 

unknown_64: $

unknown_65: 

unknown_66:$

unknown_67:	 

unknown_68: $

unknown_69: 

unknown_70:$

unknown_71:	 

unknown_72: $

unknown_73: 

unknown_74:$

unknown_75:	 

unknown_76: $

unknown_77: 

unknown_78:$

unknown_79:	 

unknown_80: $

unknown_81: 

unknown_82:$

unknown_83:

unknown_84:$

unknown_85:

unknown_86:

unknown_87:	т,

unknown_88:
identity

identity_1ѕбStatefulPartitionedCall┴
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74
unknown_75
unknown_76
unknown_77
unknown_78
unknown_79
unknown_80
unknown_81
unknown_82
unknown_83
unknown_84
unknown_85
unknown_86
unknown_87
unknown_88*f
Tin_
]2[*
Tout
2*
_collective_manager_ids
 *;
_output_shapes)
':         ж:         *|
_read_only_resource_inputs^
\Z	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_gomoku_resnet_1_layer_call_and_return_conditional_losses_10430061p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         жq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*С
_input_shapesм
¤:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
 
Д
2__inference_contract_10_3x3_layer_call_fn_10433201

inputs!
unknown: 
	unknown_0:
identityѕбStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_10_3x3_layer_call_and_return_conditional_losses_10429451w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
ш
x
L__inference_concatenate_20_layer_call_and_return_conditional_losses_10432717
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
Ќ
Ё
L__inference_contract_4_3x3_layer_call_and_return_conditional_losses_10429145

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Н
U
)__inference_skip_3_layer_call_fn_10432763
inputs_0
inputs_1
identityК
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_3_layer_call_and_return_conditional_losses_10429106h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
О
V
*__inference_skip_18_layer_call_fn_10433738
inputs_0
inputs_1
identity╚
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_18_layer_call_and_return_conditional_losses_10429871h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
Џ
є
L__inference_contract_1_5x5_layer_call_and_return_conditional_losses_10432639

inputs9
conv2d_readvariableop_resource:ђ-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:ђ*
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
ш
x
L__inference_concatenate_23_layer_call_and_return_conditional_losses_10432912
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
Ў

Щ
H__inference_value_head_layer_call_and_return_conditional_losses_10430046

inputs1
matmul_readvariableop_resource:	т,-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	т,*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:         W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         т,: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         т,
 
_user_specified_nameinputs
ќ
ё
K__inference_expand_17_5x5_layer_call_and_return_conditional_losses_10429791

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:          m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
Ь
w
M__inference_all_value_input_layer_call_and_return_conditional_losses_10429982

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :}
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         _
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         	:W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         	
 
_user_specified_nameinputs
Н
U
)__inference_skip_7_layer_call_fn_10433023
inputs_0
inputs_1
identityК
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_7_layer_call_and_return_conditional_losses_10429310h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
т
]
1__inference_concatenate_36_layer_call_fn_10433750
inputs_0
inputs_1
identity¤
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_36_layer_call_and_return_conditional_losses_10429880h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
О
V
*__inference_skip_11_layer_call_fn_10433283
inputs_0
inputs_1
identity╚
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_11_layer_call_and_return_conditional_losses_10429514h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
Ћ
Ѓ
J__inference_expand_5_5x5_layer_call_and_return_conditional_losses_10432867

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:          m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
т
]
1__inference_concatenate_28_layer_call_fn_10433230
inputs_0
inputs_1
identity¤
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_28_layer_call_and_return_conditional_losses_10429472h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
«
J
.__inference_policy_head_layer_call_fn_10433953

inputs
identityИ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ж* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_policy_head_layer_call_and_return_conditional_losses_10430057a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ж"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ж:P L
(
_output_shapes
:         ж
 
_user_specified_nameinputs
ќ
ё
K__inference_expand_10_5x5_layer_call_and_return_conditional_losses_10429434

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:          m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
О
V
*__inference_skip_14_layer_call_fn_10433478
inputs_0
inputs_1
identity╚
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_14_layer_call_and_return_conditional_losses_10429667h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
ш
x
L__inference_concatenate_37_layer_call_and_return_conditional_losses_10433822
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
Ќ
Ё
L__inference_contract_9_3x3_layer_call_and_return_conditional_losses_10429400

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Н
U
)__inference_skip_4_layer_call_fn_10432828
inputs_0
inputs_1
identityК
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_4_layer_call_and_return_conditional_losses_10429157h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
ђ
Д
1__inference_contract_1_5x5_layer_call_fn_10432628

inputs"
unknown:ђ
	unknown_0:
identityѕбStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_1_5x5_layer_call_and_return_conditional_losses_10429000w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
т
]
1__inference_concatenate_22_layer_call_fn_10432840
inputs_0
inputs_1
identity¤
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_22_layer_call_and_return_conditional_losses_10429166h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
ь
v
L__inference_concatenate_22_layer_call_and_return_conditional_losses_10429166

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :}
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
ў
є
M__inference_contract_15_3x3_layer_call_and_return_conditional_losses_10429706

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
§
д
1__inference_contract_6_3x3_layer_call_fn_10432941

inputs!
unknown: 
	unknown_0:
identityѕбStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_6_3x3_layer_call_and_return_conditional_losses_10429247w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
ш
x
L__inference_concatenate_35_layer_call_and_return_conditional_losses_10433692
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
О
V
*__inference_skip_10_layer_call_fn_10433218
inputs_0
inputs_1
identity╚
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_10_layer_call_and_return_conditional_losses_10429463h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
ѕ
Ф
5__inference_heuristic_priority_layer_call_fn_10432608

inputs"
unknown:│
	unknown_0:
identityѕбStatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_heuristic_priority_layer_call_and_return_conditional_losses_10428983w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         │: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         │
 
_user_specified_nameinputs
Ѕ
г
5__inference_heuristic_detector_layer_call_fn_10432568

inputs"
unknown:│
	unknown_0:	│
identityѕбStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         │*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_heuristic_detector_layer_call_and_return_conditional_losses_10428966x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         │`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ќ
ё
K__inference_expand_13_5x5_layer_call_and_return_conditional_losses_10433387

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:          m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
Ж
o
E__inference_skip_19_layer_call_and_return_conditional_losses_10429922

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
т
]
1__inference_concatenate_27_layer_call_fn_10433165
inputs_0
inputs_1
identity¤
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_27_layer_call_and_return_conditional_losses_10429421h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
Ы
q
E__inference_skip_18_layer_call_and_return_conditional_losses_10433744
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
щ
ц
/__inference_expand_4_5x5_layer_call_fn_10432791

inputs!
unknown:	 
	unknown_0: 
identityѕбStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_4_5x5_layer_call_and_return_conditional_losses_10429128w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
ќ
ё
K__inference_expand_18_5x5_layer_call_and_return_conditional_losses_10429842

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:          m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
т
]
1__inference_concatenate_29_layer_call_fn_10433295
inputs_0
inputs_1
identity¤
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_29_layer_call_and_return_conditional_losses_10429523h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
ќ
ё
K__inference_expand_14_5x5_layer_call_and_return_conditional_losses_10429638

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:          m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
Ћ
Ѓ
J__inference_expand_2_5x5_layer_call_and_return_conditional_losses_10429026

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:          m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
ќ
ё
K__inference_expand_12_5x5_layer_call_and_return_conditional_losses_10429536

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:          m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
Ђ
е
1__inference_expand_1_11x11_layer_call_fn_10432588

inputs"
unknown:ђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_expand_1_11x11_layer_call_and_return_conditional_losses_10428949x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
л
Џ
-__inference_value_head_layer_call_fn_10433967

inputs
unknown:	т,
	unknown_0:
identityѕбStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_value_head_layer_call_and_return_conditional_losses_10430046o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         т,: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         т,
 
_user_specified_nameinputs
ў
є
M__inference_contract_19_3x3_layer_call_and_return_conditional_losses_10433797

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Ш
y
M__inference_all_value_input_layer_call_and_return_conditional_losses_10433907
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         _
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         	:Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         	
"
_user_specified_name
inputs/1
Ж
o
E__inference_skip_15_layer_call_and_return_conditional_losses_10429718

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
Ќ
Ё
L__inference_contract_3_3x3_layer_call_and_return_conditional_losses_10432757

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
╝
J
.__inference_flat_logits_layer_call_fn_10433942

inputs
identityИ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ж* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_flat_logits_layer_call_and_return_conditional_losses_10430033a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ж"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ч
Ц
0__inference_expand_20_5x5_layer_call_fn_10433831

inputs!
unknown:	 
	unknown_0: 
identityѕбStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_20_5x5_layer_call_and_return_conditional_losses_10429944w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
ў
є
M__inference_contract_11_3x3_layer_call_and_return_conditional_losses_10429502

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
ј
ѕ
O__inference_policy_aggregator_layer_call_and_return_conditional_losses_10429995

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Ћ
Ѓ
J__inference_expand_3_5x5_layer_call_and_return_conditional_losses_10432737

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:          m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
ў
є
M__inference_contract_16_3x3_layer_call_and_return_conditional_losses_10433602

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
щ
ц
/__inference_expand_8_5x5_layer_call_fn_10433051

inputs!
unknown:	 
	unknown_0: 
identityѕбStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_8_5x5_layer_call_and_return_conditional_losses_10429332w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
Ћ
Ѓ
J__inference_expand_9_5x5_layer_call_and_return_conditional_losses_10429383

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:          m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
 
Д
2__inference_contract_12_3x3_layer_call_fn_10433331

inputs!
unknown: 
	unknown_0:
identityѕбStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_12_3x3_layer_call_and_return_conditional_losses_10429553w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
о
e
I__inference_policy_head_layer_call_and_return_conditional_losses_10433958

inputs
identityM
SoftmaxSoftmaxinputs*
T0*(
_output_shapes
:         жZ
IdentityIdentitySoftmax:softmax:0*
T0*(
_output_shapes
:         ж"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ж:P L
(
_output_shapes
:         ж
 
_user_specified_nameinputs
§
д
1__inference_contract_8_3x3_layer_call_fn_10433071

inputs!
unknown: 
	unknown_0:
identityѕбStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_8_3x3_layer_call_and_return_conditional_losses_10429349w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Ќ
І
P__inference_heuristic_detector_layer_call_and_return_conditional_losses_10428966

inputs9
conv2d_readvariableop_resource:│.
biasadd_readvariableop_resource:	│
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:│*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         │*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:│*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         │Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         │j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         │w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
т
]
1__inference_concatenate_20_layer_call_fn_10432710
inputs_0
inputs_1
identity¤
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_20_layer_call_and_return_conditional_losses_10429064h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
ќ
ё
K__inference_expand_19_5x5_layer_call_and_return_conditional_losses_10429893

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:          m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
Ж
o
E__inference_skip_17_layer_call_and_return_conditional_losses_10429820

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
§
д
1__inference_contract_5_3x3_layer_call_fn_10432876

inputs!
unknown: 
	unknown_0:
identityѕбStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_5_3x3_layer_call_and_return_conditional_losses_10429196w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
т
]
1__inference_concatenate_25_layer_call_fn_10433035
inputs_0
inputs_1
identity¤
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_25_layer_call_and_return_conditional_losses_10429319h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
Џ*
Г
2__inference_gomoku_resnet_1_layer_call_fn_10431822

inputs"
unknown:ђ
	unknown_0:	ђ$
	unknown_1:│
	unknown_2:	│$
	unknown_3:│
	unknown_4:$
	unknown_5:ђ
	unknown_6:#
	unknown_7:	 
	unknown_8: #
	unknown_9: 

unknown_10:$

unknown_11:	 

unknown_12: $

unknown_13: 

unknown_14:$

unknown_15:	 

unknown_16: $

unknown_17: 

unknown_18:$

unknown_19:	 

unknown_20: $

unknown_21: 

unknown_22:$

unknown_23:	 

unknown_24: $

unknown_25: 

unknown_26:$

unknown_27:	 

unknown_28: $

unknown_29: 

unknown_30:$

unknown_31:	 

unknown_32: $

unknown_33: 

unknown_34:$

unknown_35:	 

unknown_36: $

unknown_37: 

unknown_38:$

unknown_39:	 

unknown_40: $

unknown_41: 

unknown_42:$

unknown_43:	 

unknown_44: $

unknown_45: 

unknown_46:$

unknown_47:	 

unknown_48: $

unknown_49: 

unknown_50:$

unknown_51:	 

unknown_52: $

unknown_53: 

unknown_54:$

unknown_55:	 

unknown_56: $

unknown_57: 

unknown_58:$

unknown_59:	 

unknown_60: $

unknown_61: 

unknown_62:$

unknown_63:	 

unknown_64: $

unknown_65: 

unknown_66:$

unknown_67:	 

unknown_68: $

unknown_69: 

unknown_70:$

unknown_71:	 

unknown_72: $

unknown_73: 

unknown_74:$

unknown_75:	 

unknown_76: $

unknown_77: 

unknown_78:$

unknown_79:	 

unknown_80: $

unknown_81: 

unknown_82:$

unknown_83:

unknown_84:$

unknown_85:

unknown_86:

unknown_87:	т,

unknown_88:
identity

identity_1ѕбStatefulPartitionedCall┴
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74
unknown_75
unknown_76
unknown_77
unknown_78
unknown_79
unknown_80
unknown_81
unknown_82
unknown_83
unknown_84
unknown_85
unknown_86
unknown_87
unknown_88*f
Tin_
]2[*
Tout
2*
_collective_manager_ids
 *;
_output_shapes)
':         ж:         *|
_read_only_resource_inputs^
\Z	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_gomoku_resnet_1_layer_call_and_return_conditional_losses_10431450p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         жq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*С
_input_shapesм
¤:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ќ
ё
K__inference_expand_16_5x5_layer_call_and_return_conditional_losses_10433582

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:          m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
 
Д
2__inference_contract_11_3x3_layer_call_fn_10433266

inputs!
unknown: 
	unknown_0:
identityѕбStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_11_3x3_layer_call_and_return_conditional_losses_10429502w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Ќ
І
P__inference_heuristic_detector_layer_call_and_return_conditional_losses_10432579

inputs9
conv2d_readvariableop_resource:│.
biasadd_readvariableop_resource:	│
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:│*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         │*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:│*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         │Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         │j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         │w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
т
]
1__inference_concatenate_31_layer_call_fn_10433425
inputs_0
inputs_1
identity¤
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_31_layer_call_and_return_conditional_losses_10429625h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
нў
чa
#__inference__wrapped_model_10428931

inputsX
=gomoku_resnet_1_expand_1_11x11_conv2d_readvariableop_resource:ђM
>gomoku_resnet_1_expand_1_11x11_biasadd_readvariableop_resource:	ђ\
Agomoku_resnet_1_heuristic_detector_conv2d_readvariableop_resource:│Q
Bgomoku_resnet_1_heuristic_detector_biasadd_readvariableop_resource:	│\
Agomoku_resnet_1_heuristic_priority_conv2d_readvariableop_resource:│P
Bgomoku_resnet_1_heuristic_priority_biasadd_readvariableop_resource:X
=gomoku_resnet_1_contract_1_5x5_conv2d_readvariableop_resource:ђL
>gomoku_resnet_1_contract_1_5x5_biasadd_readvariableop_resource:U
;gomoku_resnet_1_expand_2_5x5_conv2d_readvariableop_resource:	 J
<gomoku_resnet_1_expand_2_5x5_biasadd_readvariableop_resource: W
=gomoku_resnet_1_contract_2_3x3_conv2d_readvariableop_resource: L
>gomoku_resnet_1_contract_2_3x3_biasadd_readvariableop_resource:U
;gomoku_resnet_1_expand_3_5x5_conv2d_readvariableop_resource:	 J
<gomoku_resnet_1_expand_3_5x5_biasadd_readvariableop_resource: W
=gomoku_resnet_1_contract_3_3x3_conv2d_readvariableop_resource: L
>gomoku_resnet_1_contract_3_3x3_biasadd_readvariableop_resource:U
;gomoku_resnet_1_expand_4_5x5_conv2d_readvariableop_resource:	 J
<gomoku_resnet_1_expand_4_5x5_biasadd_readvariableop_resource: W
=gomoku_resnet_1_contract_4_3x3_conv2d_readvariableop_resource: L
>gomoku_resnet_1_contract_4_3x3_biasadd_readvariableop_resource:U
;gomoku_resnet_1_expand_5_5x5_conv2d_readvariableop_resource:	 J
<gomoku_resnet_1_expand_5_5x5_biasadd_readvariableop_resource: W
=gomoku_resnet_1_contract_5_3x3_conv2d_readvariableop_resource: L
>gomoku_resnet_1_contract_5_3x3_biasadd_readvariableop_resource:U
;gomoku_resnet_1_expand_6_5x5_conv2d_readvariableop_resource:	 J
<gomoku_resnet_1_expand_6_5x5_biasadd_readvariableop_resource: W
=gomoku_resnet_1_contract_6_3x3_conv2d_readvariableop_resource: L
>gomoku_resnet_1_contract_6_3x3_biasadd_readvariableop_resource:U
;gomoku_resnet_1_expand_7_5x5_conv2d_readvariableop_resource:	 J
<gomoku_resnet_1_expand_7_5x5_biasadd_readvariableop_resource: W
=gomoku_resnet_1_contract_7_3x3_conv2d_readvariableop_resource: L
>gomoku_resnet_1_contract_7_3x3_biasadd_readvariableop_resource:U
;gomoku_resnet_1_expand_8_5x5_conv2d_readvariableop_resource:	 J
<gomoku_resnet_1_expand_8_5x5_biasadd_readvariableop_resource: W
=gomoku_resnet_1_contract_8_3x3_conv2d_readvariableop_resource: L
>gomoku_resnet_1_contract_8_3x3_biasadd_readvariableop_resource:U
;gomoku_resnet_1_expand_9_5x5_conv2d_readvariableop_resource:	 J
<gomoku_resnet_1_expand_9_5x5_biasadd_readvariableop_resource: W
=gomoku_resnet_1_contract_9_3x3_conv2d_readvariableop_resource: L
>gomoku_resnet_1_contract_9_3x3_biasadd_readvariableop_resource:V
<gomoku_resnet_1_expand_10_5x5_conv2d_readvariableop_resource:	 K
=gomoku_resnet_1_expand_10_5x5_biasadd_readvariableop_resource: X
>gomoku_resnet_1_contract_10_3x3_conv2d_readvariableop_resource: M
?gomoku_resnet_1_contract_10_3x3_biasadd_readvariableop_resource:V
<gomoku_resnet_1_expand_11_5x5_conv2d_readvariableop_resource:	 K
=gomoku_resnet_1_expand_11_5x5_biasadd_readvariableop_resource: X
>gomoku_resnet_1_contract_11_3x3_conv2d_readvariableop_resource: M
?gomoku_resnet_1_contract_11_3x3_biasadd_readvariableop_resource:V
<gomoku_resnet_1_expand_12_5x5_conv2d_readvariableop_resource:	 K
=gomoku_resnet_1_expand_12_5x5_biasadd_readvariableop_resource: X
>gomoku_resnet_1_contract_12_3x3_conv2d_readvariableop_resource: M
?gomoku_resnet_1_contract_12_3x3_biasadd_readvariableop_resource:V
<gomoku_resnet_1_expand_13_5x5_conv2d_readvariableop_resource:	 K
=gomoku_resnet_1_expand_13_5x5_biasadd_readvariableop_resource: X
>gomoku_resnet_1_contract_13_3x3_conv2d_readvariableop_resource: M
?gomoku_resnet_1_contract_13_3x3_biasadd_readvariableop_resource:V
<gomoku_resnet_1_expand_14_5x5_conv2d_readvariableop_resource:	 K
=gomoku_resnet_1_expand_14_5x5_biasadd_readvariableop_resource: X
>gomoku_resnet_1_contract_14_3x3_conv2d_readvariableop_resource: M
?gomoku_resnet_1_contract_14_3x3_biasadd_readvariableop_resource:V
<gomoku_resnet_1_expand_15_5x5_conv2d_readvariableop_resource:	 K
=gomoku_resnet_1_expand_15_5x5_biasadd_readvariableop_resource: X
>gomoku_resnet_1_contract_15_3x3_conv2d_readvariableop_resource: M
?gomoku_resnet_1_contract_15_3x3_biasadd_readvariableop_resource:V
<gomoku_resnet_1_expand_16_5x5_conv2d_readvariableop_resource:	 K
=gomoku_resnet_1_expand_16_5x5_biasadd_readvariableop_resource: X
>gomoku_resnet_1_contract_16_3x3_conv2d_readvariableop_resource: M
?gomoku_resnet_1_contract_16_3x3_biasadd_readvariableop_resource:V
<gomoku_resnet_1_expand_17_5x5_conv2d_readvariableop_resource:	 K
=gomoku_resnet_1_expand_17_5x5_biasadd_readvariableop_resource: X
>gomoku_resnet_1_contract_17_3x3_conv2d_readvariableop_resource: M
?gomoku_resnet_1_contract_17_3x3_biasadd_readvariableop_resource:V
<gomoku_resnet_1_expand_18_5x5_conv2d_readvariableop_resource:	 K
=gomoku_resnet_1_expand_18_5x5_biasadd_readvariableop_resource: X
>gomoku_resnet_1_contract_18_3x3_conv2d_readvariableop_resource: M
?gomoku_resnet_1_contract_18_3x3_biasadd_readvariableop_resource:V
<gomoku_resnet_1_expand_19_5x5_conv2d_readvariableop_resource:	 K
=gomoku_resnet_1_expand_19_5x5_biasadd_readvariableop_resource: X
>gomoku_resnet_1_contract_19_3x3_conv2d_readvariableop_resource: M
?gomoku_resnet_1_contract_19_3x3_biasadd_readvariableop_resource:V
<gomoku_resnet_1_expand_20_5x5_conv2d_readvariableop_resource:	 K
=gomoku_resnet_1_expand_20_5x5_biasadd_readvariableop_resource: X
>gomoku_resnet_1_contract_20_3x3_conv2d_readvariableop_resource: M
?gomoku_resnet_1_contract_20_3x3_biasadd_readvariableop_resource:Z
@gomoku_resnet_1_policy_aggregator_conv2d_readvariableop_resource:O
Agomoku_resnet_1_policy_aggregator_biasadd_readvariableop_resource:S
9gomoku_resnet_1_border_off_conv2d_readvariableop_resource:H
:gomoku_resnet_1_border_off_biasadd_readvariableop_resource:L
9gomoku_resnet_1_value_head_matmul_readvariableop_resource:	т,H
:gomoku_resnet_1_value_head_biasadd_readvariableop_resource:
identity

identity_1ѕб1gomoku_resnet_1/border_off/BiasAdd/ReadVariableOpб0gomoku_resnet_1/border_off/Conv2D/ReadVariableOpб6gomoku_resnet_1/contract_10_3x3/BiasAdd/ReadVariableOpб5gomoku_resnet_1/contract_10_3x3/Conv2D/ReadVariableOpб6gomoku_resnet_1/contract_11_3x3/BiasAdd/ReadVariableOpб5gomoku_resnet_1/contract_11_3x3/Conv2D/ReadVariableOpб6gomoku_resnet_1/contract_12_3x3/BiasAdd/ReadVariableOpб5gomoku_resnet_1/contract_12_3x3/Conv2D/ReadVariableOpб6gomoku_resnet_1/contract_13_3x3/BiasAdd/ReadVariableOpб5gomoku_resnet_1/contract_13_3x3/Conv2D/ReadVariableOpб6gomoku_resnet_1/contract_14_3x3/BiasAdd/ReadVariableOpб5gomoku_resnet_1/contract_14_3x3/Conv2D/ReadVariableOpб6gomoku_resnet_1/contract_15_3x3/BiasAdd/ReadVariableOpб5gomoku_resnet_1/contract_15_3x3/Conv2D/ReadVariableOpб6gomoku_resnet_1/contract_16_3x3/BiasAdd/ReadVariableOpб5gomoku_resnet_1/contract_16_3x3/Conv2D/ReadVariableOpб6gomoku_resnet_1/contract_17_3x3/BiasAdd/ReadVariableOpб5gomoku_resnet_1/contract_17_3x3/Conv2D/ReadVariableOpб6gomoku_resnet_1/contract_18_3x3/BiasAdd/ReadVariableOpб5gomoku_resnet_1/contract_18_3x3/Conv2D/ReadVariableOpб6gomoku_resnet_1/contract_19_3x3/BiasAdd/ReadVariableOpб5gomoku_resnet_1/contract_19_3x3/Conv2D/ReadVariableOpб5gomoku_resnet_1/contract_1_5x5/BiasAdd/ReadVariableOpб4gomoku_resnet_1/contract_1_5x5/Conv2D/ReadVariableOpб6gomoku_resnet_1/contract_20_3x3/BiasAdd/ReadVariableOpб5gomoku_resnet_1/contract_20_3x3/Conv2D/ReadVariableOpб5gomoku_resnet_1/contract_2_3x3/BiasAdd/ReadVariableOpб4gomoku_resnet_1/contract_2_3x3/Conv2D/ReadVariableOpб5gomoku_resnet_1/contract_3_3x3/BiasAdd/ReadVariableOpб4gomoku_resnet_1/contract_3_3x3/Conv2D/ReadVariableOpб5gomoku_resnet_1/contract_4_3x3/BiasAdd/ReadVariableOpб4gomoku_resnet_1/contract_4_3x3/Conv2D/ReadVariableOpб5gomoku_resnet_1/contract_5_3x3/BiasAdd/ReadVariableOpб4gomoku_resnet_1/contract_5_3x3/Conv2D/ReadVariableOpб5gomoku_resnet_1/contract_6_3x3/BiasAdd/ReadVariableOpб4gomoku_resnet_1/contract_6_3x3/Conv2D/ReadVariableOpб5gomoku_resnet_1/contract_7_3x3/BiasAdd/ReadVariableOpб4gomoku_resnet_1/contract_7_3x3/Conv2D/ReadVariableOpб5gomoku_resnet_1/contract_8_3x3/BiasAdd/ReadVariableOpб4gomoku_resnet_1/contract_8_3x3/Conv2D/ReadVariableOpб5gomoku_resnet_1/contract_9_3x3/BiasAdd/ReadVariableOpб4gomoku_resnet_1/contract_9_3x3/Conv2D/ReadVariableOpб4gomoku_resnet_1/expand_10_5x5/BiasAdd/ReadVariableOpб3gomoku_resnet_1/expand_10_5x5/Conv2D/ReadVariableOpб4gomoku_resnet_1/expand_11_5x5/BiasAdd/ReadVariableOpб3gomoku_resnet_1/expand_11_5x5/Conv2D/ReadVariableOpб4gomoku_resnet_1/expand_12_5x5/BiasAdd/ReadVariableOpб3gomoku_resnet_1/expand_12_5x5/Conv2D/ReadVariableOpб4gomoku_resnet_1/expand_13_5x5/BiasAdd/ReadVariableOpб3gomoku_resnet_1/expand_13_5x5/Conv2D/ReadVariableOpб4gomoku_resnet_1/expand_14_5x5/BiasAdd/ReadVariableOpб3gomoku_resnet_1/expand_14_5x5/Conv2D/ReadVariableOpб4gomoku_resnet_1/expand_15_5x5/BiasAdd/ReadVariableOpб3gomoku_resnet_1/expand_15_5x5/Conv2D/ReadVariableOpб4gomoku_resnet_1/expand_16_5x5/BiasAdd/ReadVariableOpб3gomoku_resnet_1/expand_16_5x5/Conv2D/ReadVariableOpб4gomoku_resnet_1/expand_17_5x5/BiasAdd/ReadVariableOpб3gomoku_resnet_1/expand_17_5x5/Conv2D/ReadVariableOpб4gomoku_resnet_1/expand_18_5x5/BiasAdd/ReadVariableOpб3gomoku_resnet_1/expand_18_5x5/Conv2D/ReadVariableOpб4gomoku_resnet_1/expand_19_5x5/BiasAdd/ReadVariableOpб3gomoku_resnet_1/expand_19_5x5/Conv2D/ReadVariableOpб5gomoku_resnet_1/expand_1_11x11/BiasAdd/ReadVariableOpб4gomoku_resnet_1/expand_1_11x11/Conv2D/ReadVariableOpб4gomoku_resnet_1/expand_20_5x5/BiasAdd/ReadVariableOpб3gomoku_resnet_1/expand_20_5x5/Conv2D/ReadVariableOpб3gomoku_resnet_1/expand_2_5x5/BiasAdd/ReadVariableOpб2gomoku_resnet_1/expand_2_5x5/Conv2D/ReadVariableOpб3gomoku_resnet_1/expand_3_5x5/BiasAdd/ReadVariableOpб2gomoku_resnet_1/expand_3_5x5/Conv2D/ReadVariableOpб3gomoku_resnet_1/expand_4_5x5/BiasAdd/ReadVariableOpб2gomoku_resnet_1/expand_4_5x5/Conv2D/ReadVariableOpб3gomoku_resnet_1/expand_5_5x5/BiasAdd/ReadVariableOpб2gomoku_resnet_1/expand_5_5x5/Conv2D/ReadVariableOpб3gomoku_resnet_1/expand_6_5x5/BiasAdd/ReadVariableOpб2gomoku_resnet_1/expand_6_5x5/Conv2D/ReadVariableOpб3gomoku_resnet_1/expand_7_5x5/BiasAdd/ReadVariableOpб2gomoku_resnet_1/expand_7_5x5/Conv2D/ReadVariableOpб3gomoku_resnet_1/expand_8_5x5/BiasAdd/ReadVariableOpб2gomoku_resnet_1/expand_8_5x5/Conv2D/ReadVariableOpб3gomoku_resnet_1/expand_9_5x5/BiasAdd/ReadVariableOpб2gomoku_resnet_1/expand_9_5x5/Conv2D/ReadVariableOpб9gomoku_resnet_1/heuristic_detector/BiasAdd/ReadVariableOpб8gomoku_resnet_1/heuristic_detector/Conv2D/ReadVariableOpб9gomoku_resnet_1/heuristic_priority/BiasAdd/ReadVariableOpб8gomoku_resnet_1/heuristic_priority/Conv2D/ReadVariableOpб8gomoku_resnet_1/policy_aggregator/BiasAdd/ReadVariableOpб7gomoku_resnet_1/policy_aggregator/Conv2D/ReadVariableOpб1gomoku_resnet_1/value_head/BiasAdd/ReadVariableOpб0gomoku_resnet_1/value_head/MatMul/ReadVariableOp╗
4gomoku_resnet_1/expand_1_11x11/Conv2D/ReadVariableOpReadVariableOp=gomoku_resnet_1_expand_1_11x11_conv2d_readvariableop_resource*'
_output_shapes
:ђ*
dtype0п
%gomoku_resnet_1/expand_1_11x11/Conv2DConv2Dinputs<gomoku_resnet_1/expand_1_11x11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
▒
5gomoku_resnet_1/expand_1_11x11/BiasAdd/ReadVariableOpReadVariableOp>gomoku_resnet_1_expand_1_11x11_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0█
&gomoku_resnet_1/expand_1_11x11/BiasAddBiasAdd.gomoku_resnet_1/expand_1_11x11/Conv2D:output:0=gomoku_resnet_1/expand_1_11x11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђЪ
'gomoku_resnet_1/expand_1_11x11/SoftplusSoftplus/gomoku_resnet_1/expand_1_11x11/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ├
8gomoku_resnet_1/heuristic_detector/Conv2D/ReadVariableOpReadVariableOpAgomoku_resnet_1_heuristic_detector_conv2d_readvariableop_resource*'
_output_shapes
:│*
dtype0Я
)gomoku_resnet_1/heuristic_detector/Conv2DConv2Dinputs@gomoku_resnet_1/heuristic_detector/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         │*
paddingSAME*
strides
╣
9gomoku_resnet_1/heuristic_detector/BiasAdd/ReadVariableOpReadVariableOpBgomoku_resnet_1_heuristic_detector_biasadd_readvariableop_resource*
_output_shapes	
:│*
dtype0у
*gomoku_resnet_1/heuristic_detector/BiasAddBiasAdd2gomoku_resnet_1/heuristic_detector/Conv2D:output:0Agomoku_resnet_1/heuristic_detector/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         │Ъ
'gomoku_resnet_1/heuristic_detector/ReluRelu3gomoku_resnet_1/heuristic_detector/BiasAdd:output:0*
T0*0
_output_shapes
:         │├
8gomoku_resnet_1/heuristic_priority/Conv2D/ReadVariableOpReadVariableOpAgomoku_resnet_1_heuristic_priority_conv2d_readvariableop_resource*'
_output_shapes
:│*
dtype0Ј
)gomoku_resnet_1/heuristic_priority/Conv2DConv2D5gomoku_resnet_1/heuristic_detector/Relu:activations:0@gomoku_resnet_1/heuristic_priority/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
И
9gomoku_resnet_1/heuristic_priority/BiasAdd/ReadVariableOpReadVariableOpBgomoku_resnet_1_heuristic_priority_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Т
*gomoku_resnet_1/heuristic_priority/BiasAddBiasAdd2gomoku_resnet_1/heuristic_priority/Conv2D:output:0Agomoku_resnet_1/heuristic_priority/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ъ
'gomoku_resnet_1/heuristic_priority/TanhTanh3gomoku_resnet_1/heuristic_priority/BiasAdd:output:0*
T0*/
_output_shapes
:         ╗
4gomoku_resnet_1/contract_1_5x5/Conv2D/ReadVariableOpReadVariableOp=gomoku_resnet_1_contract_1_5x5_conv2d_readvariableop_resource*'
_output_shapes
:ђ*
dtype0є
%gomoku_resnet_1/contract_1_5x5/Conv2DConv2D5gomoku_resnet_1/expand_1_11x11/Softplus:activations:0<gomoku_resnet_1/contract_1_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
░
5gomoku_resnet_1/contract_1_5x5/BiasAdd/ReadVariableOpReadVariableOp>gomoku_resnet_1_contract_1_5x5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┌
&gomoku_resnet_1/contract_1_5x5/BiasAddBiasAdd.gomoku_resnet_1/contract_1_5x5/Conv2D:output:0=gomoku_resnet_1/contract_1_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ъ
'gomoku_resnet_1/contract_1_5x5/SoftplusSoftplus/gomoku_resnet_1/contract_1_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:         l
*gomoku_resnet_1/concatenate_19/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ї
%gomoku_resnet_1/concatenate_19/concatConcatV2+gomoku_resnet_1/heuristic_priority/Tanh:y:05gomoku_resnet_1/contract_1_5x5/Softplus:activations:03gomoku_resnet_1/concatenate_19/concat/axis:output:0*
N*
T0*/
_output_shapes
:         	Х
2gomoku_resnet_1/expand_2_5x5/Conv2D/ReadVariableOpReadVariableOp;gomoku_resnet_1_expand_2_5x5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0ч
#gomoku_resnet_1/expand_2_5x5/Conv2DConv2D.gomoku_resnet_1/concatenate_19/concat:output:0:gomoku_resnet_1/expand_2_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
г
3gomoku_resnet_1/expand_2_5x5/BiasAdd/ReadVariableOpReadVariableOp<gomoku_resnet_1_expand_2_5x5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0н
$gomoku_resnet_1/expand_2_5x5/BiasAddBiasAdd,gomoku_resnet_1/expand_2_5x5/Conv2D:output:0;gomoku_resnet_1/expand_2_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          џ
%gomoku_resnet_1/expand_2_5x5/SoftplusSoftplus-gomoku_resnet_1/expand_2_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:          ║
4gomoku_resnet_1/contract_2_3x3/Conv2D/ReadVariableOpReadVariableOp=gomoku_resnet_1_contract_2_3x3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ё
%gomoku_resnet_1/contract_2_3x3/Conv2DConv2D3gomoku_resnet_1/expand_2_5x5/Softplus:activations:0<gomoku_resnet_1/contract_2_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
░
5gomoku_resnet_1/contract_2_3x3/BiasAdd/ReadVariableOpReadVariableOp>gomoku_resnet_1_contract_2_3x3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┌
&gomoku_resnet_1/contract_2_3x3/BiasAddBiasAdd.gomoku_resnet_1/contract_2_3x3/Conv2D:output:0=gomoku_resnet_1/contract_2_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ъ
'gomoku_resnet_1/contract_2_3x3/SoftplusSoftplus/gomoku_resnet_1/contract_2_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:         ╦
gomoku_resnet_1/skip_2/addAddV25gomoku_resnet_1/contract_2_3x3/Softplus:activations:05gomoku_resnet_1/contract_1_5x5/Softplus:activations:0*
T0*/
_output_shapes
:         l
*gomoku_resnet_1/concatenate_20/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ш
%gomoku_resnet_1/concatenate_20/concatConcatV2+gomoku_resnet_1/heuristic_priority/Tanh:y:0gomoku_resnet_1/skip_2/add:z:03gomoku_resnet_1/concatenate_20/concat/axis:output:0*
N*
T0*/
_output_shapes
:         	Х
2gomoku_resnet_1/expand_3_5x5/Conv2D/ReadVariableOpReadVariableOp;gomoku_resnet_1_expand_3_5x5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0ч
#gomoku_resnet_1/expand_3_5x5/Conv2DConv2D.gomoku_resnet_1/concatenate_20/concat:output:0:gomoku_resnet_1/expand_3_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
г
3gomoku_resnet_1/expand_3_5x5/BiasAdd/ReadVariableOpReadVariableOp<gomoku_resnet_1_expand_3_5x5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0н
$gomoku_resnet_1/expand_3_5x5/BiasAddBiasAdd,gomoku_resnet_1/expand_3_5x5/Conv2D:output:0;gomoku_resnet_1/expand_3_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          џ
%gomoku_resnet_1/expand_3_5x5/SoftplusSoftplus-gomoku_resnet_1/expand_3_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:          ║
4gomoku_resnet_1/contract_3_3x3/Conv2D/ReadVariableOpReadVariableOp=gomoku_resnet_1_contract_3_3x3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ё
%gomoku_resnet_1/contract_3_3x3/Conv2DConv2D3gomoku_resnet_1/expand_3_5x5/Softplus:activations:0<gomoku_resnet_1/contract_3_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
░
5gomoku_resnet_1/contract_3_3x3/BiasAdd/ReadVariableOpReadVariableOp>gomoku_resnet_1_contract_3_3x3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┌
&gomoku_resnet_1/contract_3_3x3/BiasAddBiasAdd.gomoku_resnet_1/contract_3_3x3/Conv2D:output:0=gomoku_resnet_1/contract_3_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ъ
'gomoku_resnet_1/contract_3_3x3/SoftplusSoftplus/gomoku_resnet_1/contract_3_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:         ┤
gomoku_resnet_1/skip_3/addAddV25gomoku_resnet_1/contract_3_3x3/Softplus:activations:0gomoku_resnet_1/skip_2/add:z:0*
T0*/
_output_shapes
:         l
*gomoku_resnet_1/concatenate_21/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ш
%gomoku_resnet_1/concatenate_21/concatConcatV2+gomoku_resnet_1/heuristic_priority/Tanh:y:0gomoku_resnet_1/skip_3/add:z:03gomoku_resnet_1/concatenate_21/concat/axis:output:0*
N*
T0*/
_output_shapes
:         	Х
2gomoku_resnet_1/expand_4_5x5/Conv2D/ReadVariableOpReadVariableOp;gomoku_resnet_1_expand_4_5x5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0ч
#gomoku_resnet_1/expand_4_5x5/Conv2DConv2D.gomoku_resnet_1/concatenate_21/concat:output:0:gomoku_resnet_1/expand_4_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
г
3gomoku_resnet_1/expand_4_5x5/BiasAdd/ReadVariableOpReadVariableOp<gomoku_resnet_1_expand_4_5x5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0н
$gomoku_resnet_1/expand_4_5x5/BiasAddBiasAdd,gomoku_resnet_1/expand_4_5x5/Conv2D:output:0;gomoku_resnet_1/expand_4_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          џ
%gomoku_resnet_1/expand_4_5x5/SoftplusSoftplus-gomoku_resnet_1/expand_4_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:          ║
4gomoku_resnet_1/contract_4_3x3/Conv2D/ReadVariableOpReadVariableOp=gomoku_resnet_1_contract_4_3x3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ё
%gomoku_resnet_1/contract_4_3x3/Conv2DConv2D3gomoku_resnet_1/expand_4_5x5/Softplus:activations:0<gomoku_resnet_1/contract_4_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
░
5gomoku_resnet_1/contract_4_3x3/BiasAdd/ReadVariableOpReadVariableOp>gomoku_resnet_1_contract_4_3x3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┌
&gomoku_resnet_1/contract_4_3x3/BiasAddBiasAdd.gomoku_resnet_1/contract_4_3x3/Conv2D:output:0=gomoku_resnet_1/contract_4_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ъ
'gomoku_resnet_1/contract_4_3x3/SoftplusSoftplus/gomoku_resnet_1/contract_4_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:         ┤
gomoku_resnet_1/skip_4/addAddV25gomoku_resnet_1/contract_4_3x3/Softplus:activations:0gomoku_resnet_1/skip_3/add:z:0*
T0*/
_output_shapes
:         l
*gomoku_resnet_1/concatenate_22/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ш
%gomoku_resnet_1/concatenate_22/concatConcatV2+gomoku_resnet_1/heuristic_priority/Tanh:y:0gomoku_resnet_1/skip_4/add:z:03gomoku_resnet_1/concatenate_22/concat/axis:output:0*
N*
T0*/
_output_shapes
:         	Х
2gomoku_resnet_1/expand_5_5x5/Conv2D/ReadVariableOpReadVariableOp;gomoku_resnet_1_expand_5_5x5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0ч
#gomoku_resnet_1/expand_5_5x5/Conv2DConv2D.gomoku_resnet_1/concatenate_22/concat:output:0:gomoku_resnet_1/expand_5_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
г
3gomoku_resnet_1/expand_5_5x5/BiasAdd/ReadVariableOpReadVariableOp<gomoku_resnet_1_expand_5_5x5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0н
$gomoku_resnet_1/expand_5_5x5/BiasAddBiasAdd,gomoku_resnet_1/expand_5_5x5/Conv2D:output:0;gomoku_resnet_1/expand_5_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          џ
%gomoku_resnet_1/expand_5_5x5/SoftplusSoftplus-gomoku_resnet_1/expand_5_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:          ║
4gomoku_resnet_1/contract_5_3x3/Conv2D/ReadVariableOpReadVariableOp=gomoku_resnet_1_contract_5_3x3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ё
%gomoku_resnet_1/contract_5_3x3/Conv2DConv2D3gomoku_resnet_1/expand_5_5x5/Softplus:activations:0<gomoku_resnet_1/contract_5_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
░
5gomoku_resnet_1/contract_5_3x3/BiasAdd/ReadVariableOpReadVariableOp>gomoku_resnet_1_contract_5_3x3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┌
&gomoku_resnet_1/contract_5_3x3/BiasAddBiasAdd.gomoku_resnet_1/contract_5_3x3/Conv2D:output:0=gomoku_resnet_1/contract_5_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ъ
'gomoku_resnet_1/contract_5_3x3/SoftplusSoftplus/gomoku_resnet_1/contract_5_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:         ┤
gomoku_resnet_1/skip_5/addAddV25gomoku_resnet_1/contract_5_3x3/Softplus:activations:0gomoku_resnet_1/skip_4/add:z:0*
T0*/
_output_shapes
:         l
*gomoku_resnet_1/concatenate_23/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ш
%gomoku_resnet_1/concatenate_23/concatConcatV2+gomoku_resnet_1/heuristic_priority/Tanh:y:0gomoku_resnet_1/skip_5/add:z:03gomoku_resnet_1/concatenate_23/concat/axis:output:0*
N*
T0*/
_output_shapes
:         	Х
2gomoku_resnet_1/expand_6_5x5/Conv2D/ReadVariableOpReadVariableOp;gomoku_resnet_1_expand_6_5x5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0ч
#gomoku_resnet_1/expand_6_5x5/Conv2DConv2D.gomoku_resnet_1/concatenate_23/concat:output:0:gomoku_resnet_1/expand_6_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
г
3gomoku_resnet_1/expand_6_5x5/BiasAdd/ReadVariableOpReadVariableOp<gomoku_resnet_1_expand_6_5x5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0н
$gomoku_resnet_1/expand_6_5x5/BiasAddBiasAdd,gomoku_resnet_1/expand_6_5x5/Conv2D:output:0;gomoku_resnet_1/expand_6_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          џ
%gomoku_resnet_1/expand_6_5x5/SoftplusSoftplus-gomoku_resnet_1/expand_6_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:          ║
4gomoku_resnet_1/contract_6_3x3/Conv2D/ReadVariableOpReadVariableOp=gomoku_resnet_1_contract_6_3x3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ё
%gomoku_resnet_1/contract_6_3x3/Conv2DConv2D3gomoku_resnet_1/expand_6_5x5/Softplus:activations:0<gomoku_resnet_1/contract_6_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
░
5gomoku_resnet_1/contract_6_3x3/BiasAdd/ReadVariableOpReadVariableOp>gomoku_resnet_1_contract_6_3x3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┌
&gomoku_resnet_1/contract_6_3x3/BiasAddBiasAdd.gomoku_resnet_1/contract_6_3x3/Conv2D:output:0=gomoku_resnet_1/contract_6_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ъ
'gomoku_resnet_1/contract_6_3x3/SoftplusSoftplus/gomoku_resnet_1/contract_6_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:         ┤
gomoku_resnet_1/skip_6/addAddV25gomoku_resnet_1/contract_6_3x3/Softplus:activations:0gomoku_resnet_1/skip_5/add:z:0*
T0*/
_output_shapes
:         l
*gomoku_resnet_1/concatenate_24/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ш
%gomoku_resnet_1/concatenate_24/concatConcatV2+gomoku_resnet_1/heuristic_priority/Tanh:y:0gomoku_resnet_1/skip_6/add:z:03gomoku_resnet_1/concatenate_24/concat/axis:output:0*
N*
T0*/
_output_shapes
:         	Х
2gomoku_resnet_1/expand_7_5x5/Conv2D/ReadVariableOpReadVariableOp;gomoku_resnet_1_expand_7_5x5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0ч
#gomoku_resnet_1/expand_7_5x5/Conv2DConv2D.gomoku_resnet_1/concatenate_24/concat:output:0:gomoku_resnet_1/expand_7_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
г
3gomoku_resnet_1/expand_7_5x5/BiasAdd/ReadVariableOpReadVariableOp<gomoku_resnet_1_expand_7_5x5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0н
$gomoku_resnet_1/expand_7_5x5/BiasAddBiasAdd,gomoku_resnet_1/expand_7_5x5/Conv2D:output:0;gomoku_resnet_1/expand_7_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          џ
%gomoku_resnet_1/expand_7_5x5/SoftplusSoftplus-gomoku_resnet_1/expand_7_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:          ║
4gomoku_resnet_1/contract_7_3x3/Conv2D/ReadVariableOpReadVariableOp=gomoku_resnet_1_contract_7_3x3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ё
%gomoku_resnet_1/contract_7_3x3/Conv2DConv2D3gomoku_resnet_1/expand_7_5x5/Softplus:activations:0<gomoku_resnet_1/contract_7_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
░
5gomoku_resnet_1/contract_7_3x3/BiasAdd/ReadVariableOpReadVariableOp>gomoku_resnet_1_contract_7_3x3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┌
&gomoku_resnet_1/contract_7_3x3/BiasAddBiasAdd.gomoku_resnet_1/contract_7_3x3/Conv2D:output:0=gomoku_resnet_1/contract_7_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ъ
'gomoku_resnet_1/contract_7_3x3/SoftplusSoftplus/gomoku_resnet_1/contract_7_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:         ┤
gomoku_resnet_1/skip_7/addAddV25gomoku_resnet_1/contract_7_3x3/Softplus:activations:0gomoku_resnet_1/skip_6/add:z:0*
T0*/
_output_shapes
:         l
*gomoku_resnet_1/concatenate_25/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ш
%gomoku_resnet_1/concatenate_25/concatConcatV2+gomoku_resnet_1/heuristic_priority/Tanh:y:0gomoku_resnet_1/skip_7/add:z:03gomoku_resnet_1/concatenate_25/concat/axis:output:0*
N*
T0*/
_output_shapes
:         	Х
2gomoku_resnet_1/expand_8_5x5/Conv2D/ReadVariableOpReadVariableOp;gomoku_resnet_1_expand_8_5x5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0ч
#gomoku_resnet_1/expand_8_5x5/Conv2DConv2D.gomoku_resnet_1/concatenate_25/concat:output:0:gomoku_resnet_1/expand_8_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
г
3gomoku_resnet_1/expand_8_5x5/BiasAdd/ReadVariableOpReadVariableOp<gomoku_resnet_1_expand_8_5x5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0н
$gomoku_resnet_1/expand_8_5x5/BiasAddBiasAdd,gomoku_resnet_1/expand_8_5x5/Conv2D:output:0;gomoku_resnet_1/expand_8_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          џ
%gomoku_resnet_1/expand_8_5x5/SoftplusSoftplus-gomoku_resnet_1/expand_8_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:          ║
4gomoku_resnet_1/contract_8_3x3/Conv2D/ReadVariableOpReadVariableOp=gomoku_resnet_1_contract_8_3x3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ё
%gomoku_resnet_1/contract_8_3x3/Conv2DConv2D3gomoku_resnet_1/expand_8_5x5/Softplus:activations:0<gomoku_resnet_1/contract_8_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
░
5gomoku_resnet_1/contract_8_3x3/BiasAdd/ReadVariableOpReadVariableOp>gomoku_resnet_1_contract_8_3x3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┌
&gomoku_resnet_1/contract_8_3x3/BiasAddBiasAdd.gomoku_resnet_1/contract_8_3x3/Conv2D:output:0=gomoku_resnet_1/contract_8_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ъ
'gomoku_resnet_1/contract_8_3x3/SoftplusSoftplus/gomoku_resnet_1/contract_8_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:         ┤
gomoku_resnet_1/skip_8/addAddV25gomoku_resnet_1/contract_8_3x3/Softplus:activations:0gomoku_resnet_1/skip_7/add:z:0*
T0*/
_output_shapes
:         l
*gomoku_resnet_1/concatenate_26/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ш
%gomoku_resnet_1/concatenate_26/concatConcatV2+gomoku_resnet_1/heuristic_priority/Tanh:y:0gomoku_resnet_1/skip_8/add:z:03gomoku_resnet_1/concatenate_26/concat/axis:output:0*
N*
T0*/
_output_shapes
:         	Х
2gomoku_resnet_1/expand_9_5x5/Conv2D/ReadVariableOpReadVariableOp;gomoku_resnet_1_expand_9_5x5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0ч
#gomoku_resnet_1/expand_9_5x5/Conv2DConv2D.gomoku_resnet_1/concatenate_26/concat:output:0:gomoku_resnet_1/expand_9_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
г
3gomoku_resnet_1/expand_9_5x5/BiasAdd/ReadVariableOpReadVariableOp<gomoku_resnet_1_expand_9_5x5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0н
$gomoku_resnet_1/expand_9_5x5/BiasAddBiasAdd,gomoku_resnet_1/expand_9_5x5/Conv2D:output:0;gomoku_resnet_1/expand_9_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          џ
%gomoku_resnet_1/expand_9_5x5/SoftplusSoftplus-gomoku_resnet_1/expand_9_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:          ║
4gomoku_resnet_1/contract_9_3x3/Conv2D/ReadVariableOpReadVariableOp=gomoku_resnet_1_contract_9_3x3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ё
%gomoku_resnet_1/contract_9_3x3/Conv2DConv2D3gomoku_resnet_1/expand_9_5x5/Softplus:activations:0<gomoku_resnet_1/contract_9_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
░
5gomoku_resnet_1/contract_9_3x3/BiasAdd/ReadVariableOpReadVariableOp>gomoku_resnet_1_contract_9_3x3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┌
&gomoku_resnet_1/contract_9_3x3/BiasAddBiasAdd.gomoku_resnet_1/contract_9_3x3/Conv2D:output:0=gomoku_resnet_1/contract_9_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ъ
'gomoku_resnet_1/contract_9_3x3/SoftplusSoftplus/gomoku_resnet_1/contract_9_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:         ┤
gomoku_resnet_1/skip_9/addAddV25gomoku_resnet_1/contract_9_3x3/Softplus:activations:0gomoku_resnet_1/skip_8/add:z:0*
T0*/
_output_shapes
:         l
*gomoku_resnet_1/concatenate_27/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ш
%gomoku_resnet_1/concatenate_27/concatConcatV2+gomoku_resnet_1/heuristic_priority/Tanh:y:0gomoku_resnet_1/skip_9/add:z:03gomoku_resnet_1/concatenate_27/concat/axis:output:0*
N*
T0*/
_output_shapes
:         	И
3gomoku_resnet_1/expand_10_5x5/Conv2D/ReadVariableOpReadVariableOp<gomoku_resnet_1_expand_10_5x5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0§
$gomoku_resnet_1/expand_10_5x5/Conv2DConv2D.gomoku_resnet_1/concatenate_27/concat:output:0;gomoku_resnet_1/expand_10_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
«
4gomoku_resnet_1/expand_10_5x5/BiasAdd/ReadVariableOpReadVariableOp=gomoku_resnet_1_expand_10_5x5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0О
%gomoku_resnet_1/expand_10_5x5/BiasAddBiasAdd-gomoku_resnet_1/expand_10_5x5/Conv2D:output:0<gomoku_resnet_1/expand_10_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          ю
&gomoku_resnet_1/expand_10_5x5/SoftplusSoftplus.gomoku_resnet_1/expand_10_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:          ╝
5gomoku_resnet_1/contract_10_3x3/Conv2D/ReadVariableOpReadVariableOp>gomoku_resnet_1_contract_10_3x3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Є
&gomoku_resnet_1/contract_10_3x3/Conv2DConv2D4gomoku_resnet_1/expand_10_5x5/Softplus:activations:0=gomoku_resnet_1/contract_10_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
▓
6gomoku_resnet_1/contract_10_3x3/BiasAdd/ReadVariableOpReadVariableOp?gomoku_resnet_1_contract_10_3x3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0П
'gomoku_resnet_1/contract_10_3x3/BiasAddBiasAdd/gomoku_resnet_1/contract_10_3x3/Conv2D:output:0>gomoku_resnet_1/contract_10_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         а
(gomoku_resnet_1/contract_10_3x3/SoftplusSoftplus0gomoku_resnet_1/contract_10_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:         Х
gomoku_resnet_1/skip_10/addAddV26gomoku_resnet_1/contract_10_3x3/Softplus:activations:0gomoku_resnet_1/skip_9/add:z:0*
T0*/
_output_shapes
:         l
*gomoku_resnet_1/concatenate_28/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :э
%gomoku_resnet_1/concatenate_28/concatConcatV2+gomoku_resnet_1/heuristic_priority/Tanh:y:0gomoku_resnet_1/skip_10/add:z:03gomoku_resnet_1/concatenate_28/concat/axis:output:0*
N*
T0*/
_output_shapes
:         	И
3gomoku_resnet_1/expand_11_5x5/Conv2D/ReadVariableOpReadVariableOp<gomoku_resnet_1_expand_11_5x5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0§
$gomoku_resnet_1/expand_11_5x5/Conv2DConv2D.gomoku_resnet_1/concatenate_28/concat:output:0;gomoku_resnet_1/expand_11_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
«
4gomoku_resnet_1/expand_11_5x5/BiasAdd/ReadVariableOpReadVariableOp=gomoku_resnet_1_expand_11_5x5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0О
%gomoku_resnet_1/expand_11_5x5/BiasAddBiasAdd-gomoku_resnet_1/expand_11_5x5/Conv2D:output:0<gomoku_resnet_1/expand_11_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          ю
&gomoku_resnet_1/expand_11_5x5/SoftplusSoftplus.gomoku_resnet_1/expand_11_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:          ╝
5gomoku_resnet_1/contract_11_3x3/Conv2D/ReadVariableOpReadVariableOp>gomoku_resnet_1_contract_11_3x3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Є
&gomoku_resnet_1/contract_11_3x3/Conv2DConv2D4gomoku_resnet_1/expand_11_5x5/Softplus:activations:0=gomoku_resnet_1/contract_11_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
▓
6gomoku_resnet_1/contract_11_3x3/BiasAdd/ReadVariableOpReadVariableOp?gomoku_resnet_1_contract_11_3x3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0П
'gomoku_resnet_1/contract_11_3x3/BiasAddBiasAdd/gomoku_resnet_1/contract_11_3x3/Conv2D:output:0>gomoku_resnet_1/contract_11_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         а
(gomoku_resnet_1/contract_11_3x3/SoftplusSoftplus0gomoku_resnet_1/contract_11_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:         и
gomoku_resnet_1/skip_11/addAddV26gomoku_resnet_1/contract_11_3x3/Softplus:activations:0gomoku_resnet_1/skip_10/add:z:0*
T0*/
_output_shapes
:         l
*gomoku_resnet_1/concatenate_29/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :э
%gomoku_resnet_1/concatenate_29/concatConcatV2+gomoku_resnet_1/heuristic_priority/Tanh:y:0gomoku_resnet_1/skip_11/add:z:03gomoku_resnet_1/concatenate_29/concat/axis:output:0*
N*
T0*/
_output_shapes
:         	И
3gomoku_resnet_1/expand_12_5x5/Conv2D/ReadVariableOpReadVariableOp<gomoku_resnet_1_expand_12_5x5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0§
$gomoku_resnet_1/expand_12_5x5/Conv2DConv2D.gomoku_resnet_1/concatenate_29/concat:output:0;gomoku_resnet_1/expand_12_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
«
4gomoku_resnet_1/expand_12_5x5/BiasAdd/ReadVariableOpReadVariableOp=gomoku_resnet_1_expand_12_5x5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0О
%gomoku_resnet_1/expand_12_5x5/BiasAddBiasAdd-gomoku_resnet_1/expand_12_5x5/Conv2D:output:0<gomoku_resnet_1/expand_12_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          ю
&gomoku_resnet_1/expand_12_5x5/SoftplusSoftplus.gomoku_resnet_1/expand_12_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:          ╝
5gomoku_resnet_1/contract_12_3x3/Conv2D/ReadVariableOpReadVariableOp>gomoku_resnet_1_contract_12_3x3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Є
&gomoku_resnet_1/contract_12_3x3/Conv2DConv2D4gomoku_resnet_1/expand_12_5x5/Softplus:activations:0=gomoku_resnet_1/contract_12_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
▓
6gomoku_resnet_1/contract_12_3x3/BiasAdd/ReadVariableOpReadVariableOp?gomoku_resnet_1_contract_12_3x3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0П
'gomoku_resnet_1/contract_12_3x3/BiasAddBiasAdd/gomoku_resnet_1/contract_12_3x3/Conv2D:output:0>gomoku_resnet_1/contract_12_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         а
(gomoku_resnet_1/contract_12_3x3/SoftplusSoftplus0gomoku_resnet_1/contract_12_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:         и
gomoku_resnet_1/skip_12/addAddV26gomoku_resnet_1/contract_12_3x3/Softplus:activations:0gomoku_resnet_1/skip_11/add:z:0*
T0*/
_output_shapes
:         l
*gomoku_resnet_1/concatenate_30/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :э
%gomoku_resnet_1/concatenate_30/concatConcatV2+gomoku_resnet_1/heuristic_priority/Tanh:y:0gomoku_resnet_1/skip_12/add:z:03gomoku_resnet_1/concatenate_30/concat/axis:output:0*
N*
T0*/
_output_shapes
:         	И
3gomoku_resnet_1/expand_13_5x5/Conv2D/ReadVariableOpReadVariableOp<gomoku_resnet_1_expand_13_5x5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0§
$gomoku_resnet_1/expand_13_5x5/Conv2DConv2D.gomoku_resnet_1/concatenate_30/concat:output:0;gomoku_resnet_1/expand_13_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
«
4gomoku_resnet_1/expand_13_5x5/BiasAdd/ReadVariableOpReadVariableOp=gomoku_resnet_1_expand_13_5x5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0О
%gomoku_resnet_1/expand_13_5x5/BiasAddBiasAdd-gomoku_resnet_1/expand_13_5x5/Conv2D:output:0<gomoku_resnet_1/expand_13_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          ю
&gomoku_resnet_1/expand_13_5x5/SoftplusSoftplus.gomoku_resnet_1/expand_13_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:          ╝
5gomoku_resnet_1/contract_13_3x3/Conv2D/ReadVariableOpReadVariableOp>gomoku_resnet_1_contract_13_3x3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Є
&gomoku_resnet_1/contract_13_3x3/Conv2DConv2D4gomoku_resnet_1/expand_13_5x5/Softplus:activations:0=gomoku_resnet_1/contract_13_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
▓
6gomoku_resnet_1/contract_13_3x3/BiasAdd/ReadVariableOpReadVariableOp?gomoku_resnet_1_contract_13_3x3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0П
'gomoku_resnet_1/contract_13_3x3/BiasAddBiasAdd/gomoku_resnet_1/contract_13_3x3/Conv2D:output:0>gomoku_resnet_1/contract_13_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         а
(gomoku_resnet_1/contract_13_3x3/SoftplusSoftplus0gomoku_resnet_1/contract_13_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:         и
gomoku_resnet_1/skip_13/addAddV26gomoku_resnet_1/contract_13_3x3/Softplus:activations:0gomoku_resnet_1/skip_12/add:z:0*
T0*/
_output_shapes
:         l
*gomoku_resnet_1/concatenate_31/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :э
%gomoku_resnet_1/concatenate_31/concatConcatV2+gomoku_resnet_1/heuristic_priority/Tanh:y:0gomoku_resnet_1/skip_13/add:z:03gomoku_resnet_1/concatenate_31/concat/axis:output:0*
N*
T0*/
_output_shapes
:         	И
3gomoku_resnet_1/expand_14_5x5/Conv2D/ReadVariableOpReadVariableOp<gomoku_resnet_1_expand_14_5x5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0§
$gomoku_resnet_1/expand_14_5x5/Conv2DConv2D.gomoku_resnet_1/concatenate_31/concat:output:0;gomoku_resnet_1/expand_14_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
«
4gomoku_resnet_1/expand_14_5x5/BiasAdd/ReadVariableOpReadVariableOp=gomoku_resnet_1_expand_14_5x5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0О
%gomoku_resnet_1/expand_14_5x5/BiasAddBiasAdd-gomoku_resnet_1/expand_14_5x5/Conv2D:output:0<gomoku_resnet_1/expand_14_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          ю
&gomoku_resnet_1/expand_14_5x5/SoftplusSoftplus.gomoku_resnet_1/expand_14_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:          ╝
5gomoku_resnet_1/contract_14_3x3/Conv2D/ReadVariableOpReadVariableOp>gomoku_resnet_1_contract_14_3x3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Є
&gomoku_resnet_1/contract_14_3x3/Conv2DConv2D4gomoku_resnet_1/expand_14_5x5/Softplus:activations:0=gomoku_resnet_1/contract_14_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
▓
6gomoku_resnet_1/contract_14_3x3/BiasAdd/ReadVariableOpReadVariableOp?gomoku_resnet_1_contract_14_3x3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0П
'gomoku_resnet_1/contract_14_3x3/BiasAddBiasAdd/gomoku_resnet_1/contract_14_3x3/Conv2D:output:0>gomoku_resnet_1/contract_14_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         а
(gomoku_resnet_1/contract_14_3x3/SoftplusSoftplus0gomoku_resnet_1/contract_14_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:         и
gomoku_resnet_1/skip_14/addAddV26gomoku_resnet_1/contract_14_3x3/Softplus:activations:0gomoku_resnet_1/skip_13/add:z:0*
T0*/
_output_shapes
:         l
*gomoku_resnet_1/concatenate_32/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :э
%gomoku_resnet_1/concatenate_32/concatConcatV2+gomoku_resnet_1/heuristic_priority/Tanh:y:0gomoku_resnet_1/skip_14/add:z:03gomoku_resnet_1/concatenate_32/concat/axis:output:0*
N*
T0*/
_output_shapes
:         	И
3gomoku_resnet_1/expand_15_5x5/Conv2D/ReadVariableOpReadVariableOp<gomoku_resnet_1_expand_15_5x5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0§
$gomoku_resnet_1/expand_15_5x5/Conv2DConv2D.gomoku_resnet_1/concatenate_32/concat:output:0;gomoku_resnet_1/expand_15_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
«
4gomoku_resnet_1/expand_15_5x5/BiasAdd/ReadVariableOpReadVariableOp=gomoku_resnet_1_expand_15_5x5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0О
%gomoku_resnet_1/expand_15_5x5/BiasAddBiasAdd-gomoku_resnet_1/expand_15_5x5/Conv2D:output:0<gomoku_resnet_1/expand_15_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          ю
&gomoku_resnet_1/expand_15_5x5/SoftplusSoftplus.gomoku_resnet_1/expand_15_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:          ╝
5gomoku_resnet_1/contract_15_3x3/Conv2D/ReadVariableOpReadVariableOp>gomoku_resnet_1_contract_15_3x3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Є
&gomoku_resnet_1/contract_15_3x3/Conv2DConv2D4gomoku_resnet_1/expand_15_5x5/Softplus:activations:0=gomoku_resnet_1/contract_15_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
▓
6gomoku_resnet_1/contract_15_3x3/BiasAdd/ReadVariableOpReadVariableOp?gomoku_resnet_1_contract_15_3x3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0П
'gomoku_resnet_1/contract_15_3x3/BiasAddBiasAdd/gomoku_resnet_1/contract_15_3x3/Conv2D:output:0>gomoku_resnet_1/contract_15_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         а
(gomoku_resnet_1/contract_15_3x3/SoftplusSoftplus0gomoku_resnet_1/contract_15_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:         и
gomoku_resnet_1/skip_15/addAddV26gomoku_resnet_1/contract_15_3x3/Softplus:activations:0gomoku_resnet_1/skip_14/add:z:0*
T0*/
_output_shapes
:         l
*gomoku_resnet_1/concatenate_33/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :э
%gomoku_resnet_1/concatenate_33/concatConcatV2+gomoku_resnet_1/heuristic_priority/Tanh:y:0gomoku_resnet_1/skip_15/add:z:03gomoku_resnet_1/concatenate_33/concat/axis:output:0*
N*
T0*/
_output_shapes
:         	И
3gomoku_resnet_1/expand_16_5x5/Conv2D/ReadVariableOpReadVariableOp<gomoku_resnet_1_expand_16_5x5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0§
$gomoku_resnet_1/expand_16_5x5/Conv2DConv2D.gomoku_resnet_1/concatenate_33/concat:output:0;gomoku_resnet_1/expand_16_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
«
4gomoku_resnet_1/expand_16_5x5/BiasAdd/ReadVariableOpReadVariableOp=gomoku_resnet_1_expand_16_5x5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0О
%gomoku_resnet_1/expand_16_5x5/BiasAddBiasAdd-gomoku_resnet_1/expand_16_5x5/Conv2D:output:0<gomoku_resnet_1/expand_16_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          ю
&gomoku_resnet_1/expand_16_5x5/SoftplusSoftplus.gomoku_resnet_1/expand_16_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:          ╝
5gomoku_resnet_1/contract_16_3x3/Conv2D/ReadVariableOpReadVariableOp>gomoku_resnet_1_contract_16_3x3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Є
&gomoku_resnet_1/contract_16_3x3/Conv2DConv2D4gomoku_resnet_1/expand_16_5x5/Softplus:activations:0=gomoku_resnet_1/contract_16_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
▓
6gomoku_resnet_1/contract_16_3x3/BiasAdd/ReadVariableOpReadVariableOp?gomoku_resnet_1_contract_16_3x3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0П
'gomoku_resnet_1/contract_16_3x3/BiasAddBiasAdd/gomoku_resnet_1/contract_16_3x3/Conv2D:output:0>gomoku_resnet_1/contract_16_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         а
(gomoku_resnet_1/contract_16_3x3/SoftplusSoftplus0gomoku_resnet_1/contract_16_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:         и
gomoku_resnet_1/skip_16/addAddV26gomoku_resnet_1/contract_16_3x3/Softplus:activations:0gomoku_resnet_1/skip_15/add:z:0*
T0*/
_output_shapes
:         l
*gomoku_resnet_1/concatenate_34/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :э
%gomoku_resnet_1/concatenate_34/concatConcatV2+gomoku_resnet_1/heuristic_priority/Tanh:y:0gomoku_resnet_1/skip_16/add:z:03gomoku_resnet_1/concatenate_34/concat/axis:output:0*
N*
T0*/
_output_shapes
:         	И
3gomoku_resnet_1/expand_17_5x5/Conv2D/ReadVariableOpReadVariableOp<gomoku_resnet_1_expand_17_5x5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0§
$gomoku_resnet_1/expand_17_5x5/Conv2DConv2D.gomoku_resnet_1/concatenate_34/concat:output:0;gomoku_resnet_1/expand_17_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
«
4gomoku_resnet_1/expand_17_5x5/BiasAdd/ReadVariableOpReadVariableOp=gomoku_resnet_1_expand_17_5x5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0О
%gomoku_resnet_1/expand_17_5x5/BiasAddBiasAdd-gomoku_resnet_1/expand_17_5x5/Conv2D:output:0<gomoku_resnet_1/expand_17_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          ю
&gomoku_resnet_1/expand_17_5x5/SoftplusSoftplus.gomoku_resnet_1/expand_17_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:          ╝
5gomoku_resnet_1/contract_17_3x3/Conv2D/ReadVariableOpReadVariableOp>gomoku_resnet_1_contract_17_3x3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Є
&gomoku_resnet_1/contract_17_3x3/Conv2DConv2D4gomoku_resnet_1/expand_17_5x5/Softplus:activations:0=gomoku_resnet_1/contract_17_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
▓
6gomoku_resnet_1/contract_17_3x3/BiasAdd/ReadVariableOpReadVariableOp?gomoku_resnet_1_contract_17_3x3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0П
'gomoku_resnet_1/contract_17_3x3/BiasAddBiasAdd/gomoku_resnet_1/contract_17_3x3/Conv2D:output:0>gomoku_resnet_1/contract_17_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         а
(gomoku_resnet_1/contract_17_3x3/SoftplusSoftplus0gomoku_resnet_1/contract_17_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:         и
gomoku_resnet_1/skip_17/addAddV26gomoku_resnet_1/contract_17_3x3/Softplus:activations:0gomoku_resnet_1/skip_16/add:z:0*
T0*/
_output_shapes
:         l
*gomoku_resnet_1/concatenate_35/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :э
%gomoku_resnet_1/concatenate_35/concatConcatV2+gomoku_resnet_1/heuristic_priority/Tanh:y:0gomoku_resnet_1/skip_17/add:z:03gomoku_resnet_1/concatenate_35/concat/axis:output:0*
N*
T0*/
_output_shapes
:         	И
3gomoku_resnet_1/expand_18_5x5/Conv2D/ReadVariableOpReadVariableOp<gomoku_resnet_1_expand_18_5x5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0§
$gomoku_resnet_1/expand_18_5x5/Conv2DConv2D.gomoku_resnet_1/concatenate_35/concat:output:0;gomoku_resnet_1/expand_18_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
«
4gomoku_resnet_1/expand_18_5x5/BiasAdd/ReadVariableOpReadVariableOp=gomoku_resnet_1_expand_18_5x5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0О
%gomoku_resnet_1/expand_18_5x5/BiasAddBiasAdd-gomoku_resnet_1/expand_18_5x5/Conv2D:output:0<gomoku_resnet_1/expand_18_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          ю
&gomoku_resnet_1/expand_18_5x5/SoftplusSoftplus.gomoku_resnet_1/expand_18_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:          ╝
5gomoku_resnet_1/contract_18_3x3/Conv2D/ReadVariableOpReadVariableOp>gomoku_resnet_1_contract_18_3x3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Є
&gomoku_resnet_1/contract_18_3x3/Conv2DConv2D4gomoku_resnet_1/expand_18_5x5/Softplus:activations:0=gomoku_resnet_1/contract_18_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
▓
6gomoku_resnet_1/contract_18_3x3/BiasAdd/ReadVariableOpReadVariableOp?gomoku_resnet_1_contract_18_3x3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0П
'gomoku_resnet_1/contract_18_3x3/BiasAddBiasAdd/gomoku_resnet_1/contract_18_3x3/Conv2D:output:0>gomoku_resnet_1/contract_18_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         а
(gomoku_resnet_1/contract_18_3x3/SoftplusSoftplus0gomoku_resnet_1/contract_18_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:         и
gomoku_resnet_1/skip_18/addAddV26gomoku_resnet_1/contract_18_3x3/Softplus:activations:0gomoku_resnet_1/skip_17/add:z:0*
T0*/
_output_shapes
:         l
*gomoku_resnet_1/concatenate_36/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :э
%gomoku_resnet_1/concatenate_36/concatConcatV2+gomoku_resnet_1/heuristic_priority/Tanh:y:0gomoku_resnet_1/skip_18/add:z:03gomoku_resnet_1/concatenate_36/concat/axis:output:0*
N*
T0*/
_output_shapes
:         	И
3gomoku_resnet_1/expand_19_5x5/Conv2D/ReadVariableOpReadVariableOp<gomoku_resnet_1_expand_19_5x5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0§
$gomoku_resnet_1/expand_19_5x5/Conv2DConv2D.gomoku_resnet_1/concatenate_36/concat:output:0;gomoku_resnet_1/expand_19_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
«
4gomoku_resnet_1/expand_19_5x5/BiasAdd/ReadVariableOpReadVariableOp=gomoku_resnet_1_expand_19_5x5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0О
%gomoku_resnet_1/expand_19_5x5/BiasAddBiasAdd-gomoku_resnet_1/expand_19_5x5/Conv2D:output:0<gomoku_resnet_1/expand_19_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          ю
&gomoku_resnet_1/expand_19_5x5/SoftplusSoftplus.gomoku_resnet_1/expand_19_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:          ╝
5gomoku_resnet_1/contract_19_3x3/Conv2D/ReadVariableOpReadVariableOp>gomoku_resnet_1_contract_19_3x3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Є
&gomoku_resnet_1/contract_19_3x3/Conv2DConv2D4gomoku_resnet_1/expand_19_5x5/Softplus:activations:0=gomoku_resnet_1/contract_19_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
▓
6gomoku_resnet_1/contract_19_3x3/BiasAdd/ReadVariableOpReadVariableOp?gomoku_resnet_1_contract_19_3x3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0П
'gomoku_resnet_1/contract_19_3x3/BiasAddBiasAdd/gomoku_resnet_1/contract_19_3x3/Conv2D:output:0>gomoku_resnet_1/contract_19_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         а
(gomoku_resnet_1/contract_19_3x3/SoftplusSoftplus0gomoku_resnet_1/contract_19_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:         и
gomoku_resnet_1/skip_19/addAddV26gomoku_resnet_1/contract_19_3x3/Softplus:activations:0gomoku_resnet_1/skip_18/add:z:0*
T0*/
_output_shapes
:         l
*gomoku_resnet_1/concatenate_37/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :э
%gomoku_resnet_1/concatenate_37/concatConcatV2+gomoku_resnet_1/heuristic_priority/Tanh:y:0gomoku_resnet_1/skip_19/add:z:03gomoku_resnet_1/concatenate_37/concat/axis:output:0*
N*
T0*/
_output_shapes
:         	И
3gomoku_resnet_1/expand_20_5x5/Conv2D/ReadVariableOpReadVariableOp<gomoku_resnet_1_expand_20_5x5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0§
$gomoku_resnet_1/expand_20_5x5/Conv2DConv2D.gomoku_resnet_1/concatenate_37/concat:output:0;gomoku_resnet_1/expand_20_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
«
4gomoku_resnet_1/expand_20_5x5/BiasAdd/ReadVariableOpReadVariableOp=gomoku_resnet_1_expand_20_5x5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0О
%gomoku_resnet_1/expand_20_5x5/BiasAddBiasAdd-gomoku_resnet_1/expand_20_5x5/Conv2D:output:0<gomoku_resnet_1/expand_20_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          ю
&gomoku_resnet_1/expand_20_5x5/SoftplusSoftplus.gomoku_resnet_1/expand_20_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:          ╝
5gomoku_resnet_1/contract_20_3x3/Conv2D/ReadVariableOpReadVariableOp>gomoku_resnet_1_contract_20_3x3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Є
&gomoku_resnet_1/contract_20_3x3/Conv2DConv2D4gomoku_resnet_1/expand_20_5x5/Softplus:activations:0=gomoku_resnet_1/contract_20_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
▓
6gomoku_resnet_1/contract_20_3x3/BiasAdd/ReadVariableOpReadVariableOp?gomoku_resnet_1_contract_20_3x3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0П
'gomoku_resnet_1/contract_20_3x3/BiasAddBiasAdd/gomoku_resnet_1/contract_20_3x3/Conv2D:output:0>gomoku_resnet_1/contract_20_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         а
(gomoku_resnet_1/contract_20_3x3/SoftplusSoftplus0gomoku_resnet_1/contract_20_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:         и
gomoku_resnet_1/skip_20/addAddV26gomoku_resnet_1/contract_20_3x3/Softplus:activations:0gomoku_resnet_1/skip_19/add:z:0*
T0*/
_output_shapes
:         m
+gomoku_resnet_1/all_value_input/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Њ
&gomoku_resnet_1/all_value_input/concatConcatV26gomoku_resnet_1/contract_20_3x3/Softplus:activations:0.gomoku_resnet_1/concatenate_19/concat:output:04gomoku_resnet_1/all_value_input/concat/axis:output:0*
N*
T0*/
_output_shapes
:         └
7gomoku_resnet_1/policy_aggregator/Conv2D/ReadVariableOpReadVariableOp@gomoku_resnet_1_policy_aggregator_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ш
(gomoku_resnet_1/policy_aggregator/Conv2DConv2Dgomoku_resnet_1/skip_20/add:z:0?gomoku_resnet_1/policy_aggregator/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
Х
8gomoku_resnet_1/policy_aggregator/BiasAdd/ReadVariableOpReadVariableOpAgomoku_resnet_1_policy_aggregator_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0с
)gomoku_resnet_1/policy_aggregator/BiasAddBiasAdd1gomoku_resnet_1/policy_aggregator/Conv2D:output:0@gomoku_resnet_1/policy_aggregator/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ю
&gomoku_resnet_1/policy_aggregator/ReluRelu2gomoku_resnet_1/policy_aggregator/BiasAdd:output:0*
T0*/
_output_shapes
:         w
&gomoku_resnet_1/flat_value_input/ConstConst*
_output_shapes
:*
dtype0*
valueB"    e  ╚
(gomoku_resnet_1/flat_value_input/ReshapeReshape/gomoku_resnet_1/all_value_input/concat:output:0/gomoku_resnet_1/flat_value_input/Const:output:0*
T0*(
_output_shapes
:         т,▓
0gomoku_resnet_1/border_off/Conv2D/ReadVariableOpReadVariableOp9gomoku_resnet_1_border_off_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0■
!gomoku_resnet_1/border_off/Conv2DConv2D4gomoku_resnet_1/policy_aggregator/Relu:activations:08gomoku_resnet_1/border_off/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
е
1gomoku_resnet_1/border_off/BiasAdd/ReadVariableOpReadVariableOp:gomoku_resnet_1_border_off_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╬
"gomoku_resnet_1/border_off/BiasAddBiasAdd*gomoku_resnet_1/border_off/Conv2D:output:09gomoku_resnet_1/border_off/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         p
+gomoku_resnet_1/tf.math.truediv_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ╚Bл
)gomoku_resnet_1/tf.math.truediv_1/truedivRealDiv1gomoku_resnet_1/flat_value_input/Reshape:output:04gomoku_resnet_1/tf.math.truediv_1/truediv/y:output:0*
T0*(
_output_shapes
:         т,r
!gomoku_resnet_1/flat_logits/ConstConst*
_output_shapes
:*
dtype0*
valueB"    i  ║
#gomoku_resnet_1/flat_logits/ReshapeReshape+gomoku_resnet_1/border_off/BiasAdd:output:0*gomoku_resnet_1/flat_logits/Const:output:0*
T0*(
_output_shapes
:         жФ
0gomoku_resnet_1/value_head/MatMul/ReadVariableOpReadVariableOp9gomoku_resnet_1_value_head_matmul_readvariableop_resource*
_output_shapes
:	т,*
dtype0к
!gomoku_resnet_1/value_head/MatMulMatMul-gomoku_resnet_1/tf.math.truediv_1/truediv:z:08gomoku_resnet_1/value_head/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         е
1gomoku_resnet_1/value_head/BiasAdd/ReadVariableOpReadVariableOp:gomoku_resnet_1_value_head_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0К
"gomoku_resnet_1/value_head/BiasAddBiasAdd+gomoku_resnet_1/value_head/MatMul:product:09gomoku_resnet_1/value_head/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         є
gomoku_resnet_1/value_head/TanhTanh+gomoku_resnet_1/value_head/BiasAdd:output:0*
T0*'
_output_shapes
:         Ј
#gomoku_resnet_1/policy_head/SoftmaxSoftmax,gomoku_resnet_1/flat_logits/Reshape:output:0*
T0*(
_output_shapes
:         ж}
IdentityIdentity-gomoku_resnet_1/policy_head/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:         жt

Identity_1Identity#gomoku_resnet_1/value_head/Tanh:y:0^NoOp*
T0*'
_output_shapes
:         »'
NoOpNoOp2^gomoku_resnet_1/border_off/BiasAdd/ReadVariableOp1^gomoku_resnet_1/border_off/Conv2D/ReadVariableOp7^gomoku_resnet_1/contract_10_3x3/BiasAdd/ReadVariableOp6^gomoku_resnet_1/contract_10_3x3/Conv2D/ReadVariableOp7^gomoku_resnet_1/contract_11_3x3/BiasAdd/ReadVariableOp6^gomoku_resnet_1/contract_11_3x3/Conv2D/ReadVariableOp7^gomoku_resnet_1/contract_12_3x3/BiasAdd/ReadVariableOp6^gomoku_resnet_1/contract_12_3x3/Conv2D/ReadVariableOp7^gomoku_resnet_1/contract_13_3x3/BiasAdd/ReadVariableOp6^gomoku_resnet_1/contract_13_3x3/Conv2D/ReadVariableOp7^gomoku_resnet_1/contract_14_3x3/BiasAdd/ReadVariableOp6^gomoku_resnet_1/contract_14_3x3/Conv2D/ReadVariableOp7^gomoku_resnet_1/contract_15_3x3/BiasAdd/ReadVariableOp6^gomoku_resnet_1/contract_15_3x3/Conv2D/ReadVariableOp7^gomoku_resnet_1/contract_16_3x3/BiasAdd/ReadVariableOp6^gomoku_resnet_1/contract_16_3x3/Conv2D/ReadVariableOp7^gomoku_resnet_1/contract_17_3x3/BiasAdd/ReadVariableOp6^gomoku_resnet_1/contract_17_3x3/Conv2D/ReadVariableOp7^gomoku_resnet_1/contract_18_3x3/BiasAdd/ReadVariableOp6^gomoku_resnet_1/contract_18_3x3/Conv2D/ReadVariableOp7^gomoku_resnet_1/contract_19_3x3/BiasAdd/ReadVariableOp6^gomoku_resnet_1/contract_19_3x3/Conv2D/ReadVariableOp6^gomoku_resnet_1/contract_1_5x5/BiasAdd/ReadVariableOp5^gomoku_resnet_1/contract_1_5x5/Conv2D/ReadVariableOp7^gomoku_resnet_1/contract_20_3x3/BiasAdd/ReadVariableOp6^gomoku_resnet_1/contract_20_3x3/Conv2D/ReadVariableOp6^gomoku_resnet_1/contract_2_3x3/BiasAdd/ReadVariableOp5^gomoku_resnet_1/contract_2_3x3/Conv2D/ReadVariableOp6^gomoku_resnet_1/contract_3_3x3/BiasAdd/ReadVariableOp5^gomoku_resnet_1/contract_3_3x3/Conv2D/ReadVariableOp6^gomoku_resnet_1/contract_4_3x3/BiasAdd/ReadVariableOp5^gomoku_resnet_1/contract_4_3x3/Conv2D/ReadVariableOp6^gomoku_resnet_1/contract_5_3x3/BiasAdd/ReadVariableOp5^gomoku_resnet_1/contract_5_3x3/Conv2D/ReadVariableOp6^gomoku_resnet_1/contract_6_3x3/BiasAdd/ReadVariableOp5^gomoku_resnet_1/contract_6_3x3/Conv2D/ReadVariableOp6^gomoku_resnet_1/contract_7_3x3/BiasAdd/ReadVariableOp5^gomoku_resnet_1/contract_7_3x3/Conv2D/ReadVariableOp6^gomoku_resnet_1/contract_8_3x3/BiasAdd/ReadVariableOp5^gomoku_resnet_1/contract_8_3x3/Conv2D/ReadVariableOp6^gomoku_resnet_1/contract_9_3x3/BiasAdd/ReadVariableOp5^gomoku_resnet_1/contract_9_3x3/Conv2D/ReadVariableOp5^gomoku_resnet_1/expand_10_5x5/BiasAdd/ReadVariableOp4^gomoku_resnet_1/expand_10_5x5/Conv2D/ReadVariableOp5^gomoku_resnet_1/expand_11_5x5/BiasAdd/ReadVariableOp4^gomoku_resnet_1/expand_11_5x5/Conv2D/ReadVariableOp5^gomoku_resnet_1/expand_12_5x5/BiasAdd/ReadVariableOp4^gomoku_resnet_1/expand_12_5x5/Conv2D/ReadVariableOp5^gomoku_resnet_1/expand_13_5x5/BiasAdd/ReadVariableOp4^gomoku_resnet_1/expand_13_5x5/Conv2D/ReadVariableOp5^gomoku_resnet_1/expand_14_5x5/BiasAdd/ReadVariableOp4^gomoku_resnet_1/expand_14_5x5/Conv2D/ReadVariableOp5^gomoku_resnet_1/expand_15_5x5/BiasAdd/ReadVariableOp4^gomoku_resnet_1/expand_15_5x5/Conv2D/ReadVariableOp5^gomoku_resnet_1/expand_16_5x5/BiasAdd/ReadVariableOp4^gomoku_resnet_1/expand_16_5x5/Conv2D/ReadVariableOp5^gomoku_resnet_1/expand_17_5x5/BiasAdd/ReadVariableOp4^gomoku_resnet_1/expand_17_5x5/Conv2D/ReadVariableOp5^gomoku_resnet_1/expand_18_5x5/BiasAdd/ReadVariableOp4^gomoku_resnet_1/expand_18_5x5/Conv2D/ReadVariableOp5^gomoku_resnet_1/expand_19_5x5/BiasAdd/ReadVariableOp4^gomoku_resnet_1/expand_19_5x5/Conv2D/ReadVariableOp6^gomoku_resnet_1/expand_1_11x11/BiasAdd/ReadVariableOp5^gomoku_resnet_1/expand_1_11x11/Conv2D/ReadVariableOp5^gomoku_resnet_1/expand_20_5x5/BiasAdd/ReadVariableOp4^gomoku_resnet_1/expand_20_5x5/Conv2D/ReadVariableOp4^gomoku_resnet_1/expand_2_5x5/BiasAdd/ReadVariableOp3^gomoku_resnet_1/expand_2_5x5/Conv2D/ReadVariableOp4^gomoku_resnet_1/expand_3_5x5/BiasAdd/ReadVariableOp3^gomoku_resnet_1/expand_3_5x5/Conv2D/ReadVariableOp4^gomoku_resnet_1/expand_4_5x5/BiasAdd/ReadVariableOp3^gomoku_resnet_1/expand_4_5x5/Conv2D/ReadVariableOp4^gomoku_resnet_1/expand_5_5x5/BiasAdd/ReadVariableOp3^gomoku_resnet_1/expand_5_5x5/Conv2D/ReadVariableOp4^gomoku_resnet_1/expand_6_5x5/BiasAdd/ReadVariableOp3^gomoku_resnet_1/expand_6_5x5/Conv2D/ReadVariableOp4^gomoku_resnet_1/expand_7_5x5/BiasAdd/ReadVariableOp3^gomoku_resnet_1/expand_7_5x5/Conv2D/ReadVariableOp4^gomoku_resnet_1/expand_8_5x5/BiasAdd/ReadVariableOp3^gomoku_resnet_1/expand_8_5x5/Conv2D/ReadVariableOp4^gomoku_resnet_1/expand_9_5x5/BiasAdd/ReadVariableOp3^gomoku_resnet_1/expand_9_5x5/Conv2D/ReadVariableOp:^gomoku_resnet_1/heuristic_detector/BiasAdd/ReadVariableOp9^gomoku_resnet_1/heuristic_detector/Conv2D/ReadVariableOp:^gomoku_resnet_1/heuristic_priority/BiasAdd/ReadVariableOp9^gomoku_resnet_1/heuristic_priority/Conv2D/ReadVariableOp9^gomoku_resnet_1/policy_aggregator/BiasAdd/ReadVariableOp8^gomoku_resnet_1/policy_aggregator/Conv2D/ReadVariableOp2^gomoku_resnet_1/value_head/BiasAdd/ReadVariableOp1^gomoku_resnet_1/value_head/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*С
_input_shapesм
¤:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2f
1gomoku_resnet_1/border_off/BiasAdd/ReadVariableOp1gomoku_resnet_1/border_off/BiasAdd/ReadVariableOp2d
0gomoku_resnet_1/border_off/Conv2D/ReadVariableOp0gomoku_resnet_1/border_off/Conv2D/ReadVariableOp2p
6gomoku_resnet_1/contract_10_3x3/BiasAdd/ReadVariableOp6gomoku_resnet_1/contract_10_3x3/BiasAdd/ReadVariableOp2n
5gomoku_resnet_1/contract_10_3x3/Conv2D/ReadVariableOp5gomoku_resnet_1/contract_10_3x3/Conv2D/ReadVariableOp2p
6gomoku_resnet_1/contract_11_3x3/BiasAdd/ReadVariableOp6gomoku_resnet_1/contract_11_3x3/BiasAdd/ReadVariableOp2n
5gomoku_resnet_1/contract_11_3x3/Conv2D/ReadVariableOp5gomoku_resnet_1/contract_11_3x3/Conv2D/ReadVariableOp2p
6gomoku_resnet_1/contract_12_3x3/BiasAdd/ReadVariableOp6gomoku_resnet_1/contract_12_3x3/BiasAdd/ReadVariableOp2n
5gomoku_resnet_1/contract_12_3x3/Conv2D/ReadVariableOp5gomoku_resnet_1/contract_12_3x3/Conv2D/ReadVariableOp2p
6gomoku_resnet_1/contract_13_3x3/BiasAdd/ReadVariableOp6gomoku_resnet_1/contract_13_3x3/BiasAdd/ReadVariableOp2n
5gomoku_resnet_1/contract_13_3x3/Conv2D/ReadVariableOp5gomoku_resnet_1/contract_13_3x3/Conv2D/ReadVariableOp2p
6gomoku_resnet_1/contract_14_3x3/BiasAdd/ReadVariableOp6gomoku_resnet_1/contract_14_3x3/BiasAdd/ReadVariableOp2n
5gomoku_resnet_1/contract_14_3x3/Conv2D/ReadVariableOp5gomoku_resnet_1/contract_14_3x3/Conv2D/ReadVariableOp2p
6gomoku_resnet_1/contract_15_3x3/BiasAdd/ReadVariableOp6gomoku_resnet_1/contract_15_3x3/BiasAdd/ReadVariableOp2n
5gomoku_resnet_1/contract_15_3x3/Conv2D/ReadVariableOp5gomoku_resnet_1/contract_15_3x3/Conv2D/ReadVariableOp2p
6gomoku_resnet_1/contract_16_3x3/BiasAdd/ReadVariableOp6gomoku_resnet_1/contract_16_3x3/BiasAdd/ReadVariableOp2n
5gomoku_resnet_1/contract_16_3x3/Conv2D/ReadVariableOp5gomoku_resnet_1/contract_16_3x3/Conv2D/ReadVariableOp2p
6gomoku_resnet_1/contract_17_3x3/BiasAdd/ReadVariableOp6gomoku_resnet_1/contract_17_3x3/BiasAdd/ReadVariableOp2n
5gomoku_resnet_1/contract_17_3x3/Conv2D/ReadVariableOp5gomoku_resnet_1/contract_17_3x3/Conv2D/ReadVariableOp2p
6gomoku_resnet_1/contract_18_3x3/BiasAdd/ReadVariableOp6gomoku_resnet_1/contract_18_3x3/BiasAdd/ReadVariableOp2n
5gomoku_resnet_1/contract_18_3x3/Conv2D/ReadVariableOp5gomoku_resnet_1/contract_18_3x3/Conv2D/ReadVariableOp2p
6gomoku_resnet_1/contract_19_3x3/BiasAdd/ReadVariableOp6gomoku_resnet_1/contract_19_3x3/BiasAdd/ReadVariableOp2n
5gomoku_resnet_1/contract_19_3x3/Conv2D/ReadVariableOp5gomoku_resnet_1/contract_19_3x3/Conv2D/ReadVariableOp2n
5gomoku_resnet_1/contract_1_5x5/BiasAdd/ReadVariableOp5gomoku_resnet_1/contract_1_5x5/BiasAdd/ReadVariableOp2l
4gomoku_resnet_1/contract_1_5x5/Conv2D/ReadVariableOp4gomoku_resnet_1/contract_1_5x5/Conv2D/ReadVariableOp2p
6gomoku_resnet_1/contract_20_3x3/BiasAdd/ReadVariableOp6gomoku_resnet_1/contract_20_3x3/BiasAdd/ReadVariableOp2n
5gomoku_resnet_1/contract_20_3x3/Conv2D/ReadVariableOp5gomoku_resnet_1/contract_20_3x3/Conv2D/ReadVariableOp2n
5gomoku_resnet_1/contract_2_3x3/BiasAdd/ReadVariableOp5gomoku_resnet_1/contract_2_3x3/BiasAdd/ReadVariableOp2l
4gomoku_resnet_1/contract_2_3x3/Conv2D/ReadVariableOp4gomoku_resnet_1/contract_2_3x3/Conv2D/ReadVariableOp2n
5gomoku_resnet_1/contract_3_3x3/BiasAdd/ReadVariableOp5gomoku_resnet_1/contract_3_3x3/BiasAdd/ReadVariableOp2l
4gomoku_resnet_1/contract_3_3x3/Conv2D/ReadVariableOp4gomoku_resnet_1/contract_3_3x3/Conv2D/ReadVariableOp2n
5gomoku_resnet_1/contract_4_3x3/BiasAdd/ReadVariableOp5gomoku_resnet_1/contract_4_3x3/BiasAdd/ReadVariableOp2l
4gomoku_resnet_1/contract_4_3x3/Conv2D/ReadVariableOp4gomoku_resnet_1/contract_4_3x3/Conv2D/ReadVariableOp2n
5gomoku_resnet_1/contract_5_3x3/BiasAdd/ReadVariableOp5gomoku_resnet_1/contract_5_3x3/BiasAdd/ReadVariableOp2l
4gomoku_resnet_1/contract_5_3x3/Conv2D/ReadVariableOp4gomoku_resnet_1/contract_5_3x3/Conv2D/ReadVariableOp2n
5gomoku_resnet_1/contract_6_3x3/BiasAdd/ReadVariableOp5gomoku_resnet_1/contract_6_3x3/BiasAdd/ReadVariableOp2l
4gomoku_resnet_1/contract_6_3x3/Conv2D/ReadVariableOp4gomoku_resnet_1/contract_6_3x3/Conv2D/ReadVariableOp2n
5gomoku_resnet_1/contract_7_3x3/BiasAdd/ReadVariableOp5gomoku_resnet_1/contract_7_3x3/BiasAdd/ReadVariableOp2l
4gomoku_resnet_1/contract_7_3x3/Conv2D/ReadVariableOp4gomoku_resnet_1/contract_7_3x3/Conv2D/ReadVariableOp2n
5gomoku_resnet_1/contract_8_3x3/BiasAdd/ReadVariableOp5gomoku_resnet_1/contract_8_3x3/BiasAdd/ReadVariableOp2l
4gomoku_resnet_1/contract_8_3x3/Conv2D/ReadVariableOp4gomoku_resnet_1/contract_8_3x3/Conv2D/ReadVariableOp2n
5gomoku_resnet_1/contract_9_3x3/BiasAdd/ReadVariableOp5gomoku_resnet_1/contract_9_3x3/BiasAdd/ReadVariableOp2l
4gomoku_resnet_1/contract_9_3x3/Conv2D/ReadVariableOp4gomoku_resnet_1/contract_9_3x3/Conv2D/ReadVariableOp2l
4gomoku_resnet_1/expand_10_5x5/BiasAdd/ReadVariableOp4gomoku_resnet_1/expand_10_5x5/BiasAdd/ReadVariableOp2j
3gomoku_resnet_1/expand_10_5x5/Conv2D/ReadVariableOp3gomoku_resnet_1/expand_10_5x5/Conv2D/ReadVariableOp2l
4gomoku_resnet_1/expand_11_5x5/BiasAdd/ReadVariableOp4gomoku_resnet_1/expand_11_5x5/BiasAdd/ReadVariableOp2j
3gomoku_resnet_1/expand_11_5x5/Conv2D/ReadVariableOp3gomoku_resnet_1/expand_11_5x5/Conv2D/ReadVariableOp2l
4gomoku_resnet_1/expand_12_5x5/BiasAdd/ReadVariableOp4gomoku_resnet_1/expand_12_5x5/BiasAdd/ReadVariableOp2j
3gomoku_resnet_1/expand_12_5x5/Conv2D/ReadVariableOp3gomoku_resnet_1/expand_12_5x5/Conv2D/ReadVariableOp2l
4gomoku_resnet_1/expand_13_5x5/BiasAdd/ReadVariableOp4gomoku_resnet_1/expand_13_5x5/BiasAdd/ReadVariableOp2j
3gomoku_resnet_1/expand_13_5x5/Conv2D/ReadVariableOp3gomoku_resnet_1/expand_13_5x5/Conv2D/ReadVariableOp2l
4gomoku_resnet_1/expand_14_5x5/BiasAdd/ReadVariableOp4gomoku_resnet_1/expand_14_5x5/BiasAdd/ReadVariableOp2j
3gomoku_resnet_1/expand_14_5x5/Conv2D/ReadVariableOp3gomoku_resnet_1/expand_14_5x5/Conv2D/ReadVariableOp2l
4gomoku_resnet_1/expand_15_5x5/BiasAdd/ReadVariableOp4gomoku_resnet_1/expand_15_5x5/BiasAdd/ReadVariableOp2j
3gomoku_resnet_1/expand_15_5x5/Conv2D/ReadVariableOp3gomoku_resnet_1/expand_15_5x5/Conv2D/ReadVariableOp2l
4gomoku_resnet_1/expand_16_5x5/BiasAdd/ReadVariableOp4gomoku_resnet_1/expand_16_5x5/BiasAdd/ReadVariableOp2j
3gomoku_resnet_1/expand_16_5x5/Conv2D/ReadVariableOp3gomoku_resnet_1/expand_16_5x5/Conv2D/ReadVariableOp2l
4gomoku_resnet_1/expand_17_5x5/BiasAdd/ReadVariableOp4gomoku_resnet_1/expand_17_5x5/BiasAdd/ReadVariableOp2j
3gomoku_resnet_1/expand_17_5x5/Conv2D/ReadVariableOp3gomoku_resnet_1/expand_17_5x5/Conv2D/ReadVariableOp2l
4gomoku_resnet_1/expand_18_5x5/BiasAdd/ReadVariableOp4gomoku_resnet_1/expand_18_5x5/BiasAdd/ReadVariableOp2j
3gomoku_resnet_1/expand_18_5x5/Conv2D/ReadVariableOp3gomoku_resnet_1/expand_18_5x5/Conv2D/ReadVariableOp2l
4gomoku_resnet_1/expand_19_5x5/BiasAdd/ReadVariableOp4gomoku_resnet_1/expand_19_5x5/BiasAdd/ReadVariableOp2j
3gomoku_resnet_1/expand_19_5x5/Conv2D/ReadVariableOp3gomoku_resnet_1/expand_19_5x5/Conv2D/ReadVariableOp2n
5gomoku_resnet_1/expand_1_11x11/BiasAdd/ReadVariableOp5gomoku_resnet_1/expand_1_11x11/BiasAdd/ReadVariableOp2l
4gomoku_resnet_1/expand_1_11x11/Conv2D/ReadVariableOp4gomoku_resnet_1/expand_1_11x11/Conv2D/ReadVariableOp2l
4gomoku_resnet_1/expand_20_5x5/BiasAdd/ReadVariableOp4gomoku_resnet_1/expand_20_5x5/BiasAdd/ReadVariableOp2j
3gomoku_resnet_1/expand_20_5x5/Conv2D/ReadVariableOp3gomoku_resnet_1/expand_20_5x5/Conv2D/ReadVariableOp2j
3gomoku_resnet_1/expand_2_5x5/BiasAdd/ReadVariableOp3gomoku_resnet_1/expand_2_5x5/BiasAdd/ReadVariableOp2h
2gomoku_resnet_1/expand_2_5x5/Conv2D/ReadVariableOp2gomoku_resnet_1/expand_2_5x5/Conv2D/ReadVariableOp2j
3gomoku_resnet_1/expand_3_5x5/BiasAdd/ReadVariableOp3gomoku_resnet_1/expand_3_5x5/BiasAdd/ReadVariableOp2h
2gomoku_resnet_1/expand_3_5x5/Conv2D/ReadVariableOp2gomoku_resnet_1/expand_3_5x5/Conv2D/ReadVariableOp2j
3gomoku_resnet_1/expand_4_5x5/BiasAdd/ReadVariableOp3gomoku_resnet_1/expand_4_5x5/BiasAdd/ReadVariableOp2h
2gomoku_resnet_1/expand_4_5x5/Conv2D/ReadVariableOp2gomoku_resnet_1/expand_4_5x5/Conv2D/ReadVariableOp2j
3gomoku_resnet_1/expand_5_5x5/BiasAdd/ReadVariableOp3gomoku_resnet_1/expand_5_5x5/BiasAdd/ReadVariableOp2h
2gomoku_resnet_1/expand_5_5x5/Conv2D/ReadVariableOp2gomoku_resnet_1/expand_5_5x5/Conv2D/ReadVariableOp2j
3gomoku_resnet_1/expand_6_5x5/BiasAdd/ReadVariableOp3gomoku_resnet_1/expand_6_5x5/BiasAdd/ReadVariableOp2h
2gomoku_resnet_1/expand_6_5x5/Conv2D/ReadVariableOp2gomoku_resnet_1/expand_6_5x5/Conv2D/ReadVariableOp2j
3gomoku_resnet_1/expand_7_5x5/BiasAdd/ReadVariableOp3gomoku_resnet_1/expand_7_5x5/BiasAdd/ReadVariableOp2h
2gomoku_resnet_1/expand_7_5x5/Conv2D/ReadVariableOp2gomoku_resnet_1/expand_7_5x5/Conv2D/ReadVariableOp2j
3gomoku_resnet_1/expand_8_5x5/BiasAdd/ReadVariableOp3gomoku_resnet_1/expand_8_5x5/BiasAdd/ReadVariableOp2h
2gomoku_resnet_1/expand_8_5x5/Conv2D/ReadVariableOp2gomoku_resnet_1/expand_8_5x5/Conv2D/ReadVariableOp2j
3gomoku_resnet_1/expand_9_5x5/BiasAdd/ReadVariableOp3gomoku_resnet_1/expand_9_5x5/BiasAdd/ReadVariableOp2h
2gomoku_resnet_1/expand_9_5x5/Conv2D/ReadVariableOp2gomoku_resnet_1/expand_9_5x5/Conv2D/ReadVariableOp2v
9gomoku_resnet_1/heuristic_detector/BiasAdd/ReadVariableOp9gomoku_resnet_1/heuristic_detector/BiasAdd/ReadVariableOp2t
8gomoku_resnet_1/heuristic_detector/Conv2D/ReadVariableOp8gomoku_resnet_1/heuristic_detector/Conv2D/ReadVariableOp2v
9gomoku_resnet_1/heuristic_priority/BiasAdd/ReadVariableOp9gomoku_resnet_1/heuristic_priority/BiasAdd/ReadVariableOp2t
8gomoku_resnet_1/heuristic_priority/Conv2D/ReadVariableOp8gomoku_resnet_1/heuristic_priority/Conv2D/ReadVariableOp2t
8gomoku_resnet_1/policy_aggregator/BiasAdd/ReadVariableOp8gomoku_resnet_1/policy_aggregator/BiasAdd/ReadVariableOp2r
7gomoku_resnet_1/policy_aggregator/Conv2D/ReadVariableOp7gomoku_resnet_1/policy_aggregator/Conv2D/ReadVariableOp2f
1gomoku_resnet_1/value_head/BiasAdd/ReadVariableOp1gomoku_resnet_1/value_head/BiasAdd/ReadVariableOp2d
0gomoku_resnet_1/value_head/MatMul/ReadVariableOp0gomoku_resnet_1/value_head/MatMul/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ь
v
L__inference_concatenate_27_layer_call_and_return_conditional_losses_10429421

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :}
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
ы
p
D__inference_skip_7_layer_call_and_return_conditional_losses_10433029
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
ј
ѕ
O__inference_policy_aggregator_layer_call_and_return_conditional_losses_10433894

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╦
e
I__inference_flat_logits_layer_call_and_return_conditional_losses_10430033

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    i  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         жY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ж"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
▒т
ќ<
$__inference__traced_restore_10434564
file_prefixE
*assignvariableop_heuristic_detector_kernel:│9
*assignvariableop_1_heuristic_detector_bias:	│C
(assignvariableop_2_expand_1_11x11_kernel:ђ5
&assignvariableop_3_expand_1_11x11_bias:	ђG
,assignvariableop_4_heuristic_priority_kernel:│8
*assignvariableop_5_heuristic_priority_bias:C
(assignvariableop_6_contract_1_5x5_kernel:ђ4
&assignvariableop_7_contract_1_5x5_bias:@
&assignvariableop_8_expand_2_5x5_kernel:	 2
$assignvariableop_9_expand_2_5x5_bias: C
)assignvariableop_10_contract_2_3x3_kernel: 5
'assignvariableop_11_contract_2_3x3_bias:A
'assignvariableop_12_expand_3_5x5_kernel:	 3
%assignvariableop_13_expand_3_5x5_bias: C
)assignvariableop_14_contract_3_3x3_kernel: 5
'assignvariableop_15_contract_3_3x3_bias:A
'assignvariableop_16_expand_4_5x5_kernel:	 3
%assignvariableop_17_expand_4_5x5_bias: C
)assignvariableop_18_contract_4_3x3_kernel: 5
'assignvariableop_19_contract_4_3x3_bias:A
'assignvariableop_20_expand_5_5x5_kernel:	 3
%assignvariableop_21_expand_5_5x5_bias: C
)assignvariableop_22_contract_5_3x3_kernel: 5
'assignvariableop_23_contract_5_3x3_bias:A
'assignvariableop_24_expand_6_5x5_kernel:	 3
%assignvariableop_25_expand_6_5x5_bias: C
)assignvariableop_26_contract_6_3x3_kernel: 5
'assignvariableop_27_contract_6_3x3_bias:A
'assignvariableop_28_expand_7_5x5_kernel:	 3
%assignvariableop_29_expand_7_5x5_bias: C
)assignvariableop_30_contract_7_3x3_kernel: 5
'assignvariableop_31_contract_7_3x3_bias:A
'assignvariableop_32_expand_8_5x5_kernel:	 3
%assignvariableop_33_expand_8_5x5_bias: C
)assignvariableop_34_contract_8_3x3_kernel: 5
'assignvariableop_35_contract_8_3x3_bias:A
'assignvariableop_36_expand_9_5x5_kernel:	 3
%assignvariableop_37_expand_9_5x5_bias: C
)assignvariableop_38_contract_9_3x3_kernel: 5
'assignvariableop_39_contract_9_3x3_bias:B
(assignvariableop_40_expand_10_5x5_kernel:	 4
&assignvariableop_41_expand_10_5x5_bias: D
*assignvariableop_42_contract_10_3x3_kernel: 6
(assignvariableop_43_contract_10_3x3_bias:B
(assignvariableop_44_expand_11_5x5_kernel:	 4
&assignvariableop_45_expand_11_5x5_bias: D
*assignvariableop_46_contract_11_3x3_kernel: 6
(assignvariableop_47_contract_11_3x3_bias:B
(assignvariableop_48_expand_12_5x5_kernel:	 4
&assignvariableop_49_expand_12_5x5_bias: D
*assignvariableop_50_contract_12_3x3_kernel: 6
(assignvariableop_51_contract_12_3x3_bias:B
(assignvariableop_52_expand_13_5x5_kernel:	 4
&assignvariableop_53_expand_13_5x5_bias: D
*assignvariableop_54_contract_13_3x3_kernel: 6
(assignvariableop_55_contract_13_3x3_bias:B
(assignvariableop_56_expand_14_5x5_kernel:	 4
&assignvariableop_57_expand_14_5x5_bias: D
*assignvariableop_58_contract_14_3x3_kernel: 6
(assignvariableop_59_contract_14_3x3_bias:B
(assignvariableop_60_expand_15_5x5_kernel:	 4
&assignvariableop_61_expand_15_5x5_bias: D
*assignvariableop_62_contract_15_3x3_kernel: 6
(assignvariableop_63_contract_15_3x3_bias:B
(assignvariableop_64_expand_16_5x5_kernel:	 4
&assignvariableop_65_expand_16_5x5_bias: D
*assignvariableop_66_contract_16_3x3_kernel: 6
(assignvariableop_67_contract_16_3x3_bias:B
(assignvariableop_68_expand_17_5x5_kernel:	 4
&assignvariableop_69_expand_17_5x5_bias: D
*assignvariableop_70_contract_17_3x3_kernel: 6
(assignvariableop_71_contract_17_3x3_bias:B
(assignvariableop_72_expand_18_5x5_kernel:	 4
&assignvariableop_73_expand_18_5x5_bias: D
*assignvariableop_74_contract_18_3x3_kernel: 6
(assignvariableop_75_contract_18_3x3_bias:B
(assignvariableop_76_expand_19_5x5_kernel:	 4
&assignvariableop_77_expand_19_5x5_bias: D
*assignvariableop_78_contract_19_3x3_kernel: 6
(assignvariableop_79_contract_19_3x3_bias:B
(assignvariableop_80_expand_20_5x5_kernel:	 4
&assignvariableop_81_expand_20_5x5_bias: D
*assignvariableop_82_contract_20_3x3_kernel: 6
(assignvariableop_83_contract_20_3x3_bias:F
,assignvariableop_84_policy_aggregator_kernel:8
*assignvariableop_85_policy_aggregator_bias:?
%assignvariableop_86_border_off_kernel:1
#assignvariableop_87_border_off_bias:8
%assignvariableop_88_value_head_kernel:	т,1
#assignvariableop_89_value_head_bias:#
assignvariableop_90_total: #
assignvariableop_91_count: 
identity_93ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_39бAssignVariableOp_4бAssignVariableOp_40бAssignVariableOp_41бAssignVariableOp_42бAssignVariableOp_43бAssignVariableOp_44бAssignVariableOp_45бAssignVariableOp_46бAssignVariableOp_47бAssignVariableOp_48бAssignVariableOp_49бAssignVariableOp_5бAssignVariableOp_50бAssignVariableOp_51бAssignVariableOp_52бAssignVariableOp_53бAssignVariableOp_54бAssignVariableOp_55бAssignVariableOp_56бAssignVariableOp_57бAssignVariableOp_58бAssignVariableOp_59бAssignVariableOp_6бAssignVariableOp_60бAssignVariableOp_61бAssignVariableOp_62бAssignVariableOp_63бAssignVariableOp_64бAssignVariableOp_65бAssignVariableOp_66бAssignVariableOp_67бAssignVariableOp_68бAssignVariableOp_69бAssignVariableOp_7бAssignVariableOp_70бAssignVariableOp_71бAssignVariableOp_72бAssignVariableOp_73бAssignVariableOp_74бAssignVariableOp_75бAssignVariableOp_76бAssignVariableOp_77бAssignVariableOp_78бAssignVariableOp_79бAssignVariableOp_8бAssignVariableOp_80бAssignVariableOp_81бAssignVariableOp_82бAssignVariableOp_83бAssignVariableOp_84бAssignVariableOp_85бAssignVariableOp_86бAssignVariableOp_87бAssignVariableOp_88бAssignVariableOp_89бAssignVariableOp_9бAssignVariableOp_90бAssignVariableOp_91Ћ)
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:]*
dtype0*╗(
value▒(B«(]B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-25/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-25/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-26/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-27/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-27/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-28/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-28/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-29/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-29/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-30/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-30/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-31/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-31/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-32/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-32/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-33/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-33/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-34/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-34/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-35/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-35/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-36/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-36/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-37/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-37/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-38/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-38/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-39/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-39/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-40/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-40/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-41/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-41/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-42/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-42/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-43/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-43/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-44/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-44/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHГ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:]*
dtype0*¤
value┼B┬]B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ы
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*і
_output_shapesэ
З:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*k
dtypesa
_2][
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOpAssignVariableOp*assignvariableop_heuristic_detector_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_1AssignVariableOp*assignvariableop_1_heuristic_detector_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_2AssignVariableOp(assignvariableop_2_expand_1_11x11_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_3AssignVariableOp&assignvariableop_3_expand_1_11x11_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_4AssignVariableOp,assignvariableop_4_heuristic_priority_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_5AssignVariableOp*assignvariableop_5_heuristic_priority_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_6AssignVariableOp(assignvariableop_6_contract_1_5x5_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_7AssignVariableOp&assignvariableop_7_contract_1_5x5_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_8AssignVariableOp&assignvariableop_8_expand_2_5x5_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Њ
AssignVariableOp_9AssignVariableOp$assignvariableop_9_expand_2_5x5_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_10AssignVariableOp)assignvariableop_10_contract_2_3x3_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:ў
AssignVariableOp_11AssignVariableOp'assignvariableop_11_contract_2_3x3_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:ў
AssignVariableOp_12AssignVariableOp'assignvariableop_12_expand_3_5x5_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:ќ
AssignVariableOp_13AssignVariableOp%assignvariableop_13_expand_3_5x5_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_14AssignVariableOp)assignvariableop_14_contract_3_3x3_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:ў
AssignVariableOp_15AssignVariableOp'assignvariableop_15_contract_3_3x3_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:ў
AssignVariableOp_16AssignVariableOp'assignvariableop_16_expand_4_5x5_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:ќ
AssignVariableOp_17AssignVariableOp%assignvariableop_17_expand_4_5x5_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_18AssignVariableOp)assignvariableop_18_contract_4_3x3_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:ў
AssignVariableOp_19AssignVariableOp'assignvariableop_19_contract_4_3x3_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:ў
AssignVariableOp_20AssignVariableOp'assignvariableop_20_expand_5_5x5_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:ќ
AssignVariableOp_21AssignVariableOp%assignvariableop_21_expand_5_5x5_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_22AssignVariableOp)assignvariableop_22_contract_5_3x3_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:ў
AssignVariableOp_23AssignVariableOp'assignvariableop_23_contract_5_3x3_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:ў
AssignVariableOp_24AssignVariableOp'assignvariableop_24_expand_6_5x5_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:ќ
AssignVariableOp_25AssignVariableOp%assignvariableop_25_expand_6_5x5_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_26AssignVariableOp)assignvariableop_26_contract_6_3x3_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:ў
AssignVariableOp_27AssignVariableOp'assignvariableop_27_contract_6_3x3_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:ў
AssignVariableOp_28AssignVariableOp'assignvariableop_28_expand_7_5x5_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:ќ
AssignVariableOp_29AssignVariableOp%assignvariableop_29_expand_7_5x5_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_30AssignVariableOp)assignvariableop_30_contract_7_3x3_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:ў
AssignVariableOp_31AssignVariableOp'assignvariableop_31_contract_7_3x3_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:ў
AssignVariableOp_32AssignVariableOp'assignvariableop_32_expand_8_5x5_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:ќ
AssignVariableOp_33AssignVariableOp%assignvariableop_33_expand_8_5x5_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_34AssignVariableOp)assignvariableop_34_contract_8_3x3_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:ў
AssignVariableOp_35AssignVariableOp'assignvariableop_35_contract_8_3x3_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:ў
AssignVariableOp_36AssignVariableOp'assignvariableop_36_expand_9_5x5_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:ќ
AssignVariableOp_37AssignVariableOp%assignvariableop_37_expand_9_5x5_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_38AssignVariableOp)assignvariableop_38_contract_9_3x3_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:ў
AssignVariableOp_39AssignVariableOp'assignvariableop_39_contract_9_3x3_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_40AssignVariableOp(assignvariableop_40_expand_10_5x5_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_41AssignVariableOp&assignvariableop_41_expand_10_5x5_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_42AssignVariableOp*assignvariableop_42_contract_10_3x3_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_43AssignVariableOp(assignvariableop_43_contract_10_3x3_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_44AssignVariableOp(assignvariableop_44_expand_11_5x5_kernelIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_45AssignVariableOp&assignvariableop_45_expand_11_5x5_biasIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_46AssignVariableOp*assignvariableop_46_contract_11_3x3_kernelIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_47AssignVariableOp(assignvariableop_47_contract_11_3x3_biasIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_48AssignVariableOp(assignvariableop_48_expand_12_5x5_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_49AssignVariableOp&assignvariableop_49_expand_12_5x5_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_50AssignVariableOp*assignvariableop_50_contract_12_3x3_kernelIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_51AssignVariableOp(assignvariableop_51_contract_12_3x3_biasIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_52AssignVariableOp(assignvariableop_52_expand_13_5x5_kernelIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_53AssignVariableOp&assignvariableop_53_expand_13_5x5_biasIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_54AssignVariableOp*assignvariableop_54_contract_13_3x3_kernelIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_55AssignVariableOp(assignvariableop_55_contract_13_3x3_biasIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_56AssignVariableOp(assignvariableop_56_expand_14_5x5_kernelIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_57AssignVariableOp&assignvariableop_57_expand_14_5x5_biasIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_58AssignVariableOp*assignvariableop_58_contract_14_3x3_kernelIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_59AssignVariableOp(assignvariableop_59_contract_14_3x3_biasIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_60AssignVariableOp(assignvariableop_60_expand_15_5x5_kernelIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_61AssignVariableOp&assignvariableop_61_expand_15_5x5_biasIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_62AssignVariableOp*assignvariableop_62_contract_15_3x3_kernelIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_63AssignVariableOp(assignvariableop_63_contract_15_3x3_biasIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_64AssignVariableOp(assignvariableop_64_expand_16_5x5_kernelIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_65AssignVariableOp&assignvariableop_65_expand_16_5x5_biasIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_66AssignVariableOp*assignvariableop_66_contract_16_3x3_kernelIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_67AssignVariableOp(assignvariableop_67_contract_16_3x3_biasIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_68AssignVariableOp(assignvariableop_68_expand_17_5x5_kernelIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_69AssignVariableOp&assignvariableop_69_expand_17_5x5_biasIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_70AssignVariableOp*assignvariableop_70_contract_17_3x3_kernelIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_71AssignVariableOp(assignvariableop_71_contract_17_3x3_biasIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_72AssignVariableOp(assignvariableop_72_expand_18_5x5_kernelIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_73AssignVariableOp&assignvariableop_73_expand_18_5x5_biasIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_74AssignVariableOp*assignvariableop_74_contract_18_3x3_kernelIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_75AssignVariableOp(assignvariableop_75_contract_18_3x3_biasIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_76AssignVariableOp(assignvariableop_76_expand_19_5x5_kernelIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_77AssignVariableOp&assignvariableop_77_expand_19_5x5_biasIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_78AssignVariableOp*assignvariableop_78_contract_19_3x3_kernelIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_79AssignVariableOp(assignvariableop_79_contract_19_3x3_biasIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_80AssignVariableOp(assignvariableop_80_expand_20_5x5_kernelIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_81AssignVariableOp&assignvariableop_81_expand_20_5x5_biasIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_82AssignVariableOp*assignvariableop_82_contract_20_3x3_kernelIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_83AssignVariableOp(assignvariableop_83_contract_20_3x3_biasIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_84AssignVariableOp,assignvariableop_84_policy_aggregator_kernelIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_85AssignVariableOp*assignvariableop_85_policy_aggregator_biasIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:ќ
AssignVariableOp_86AssignVariableOp%assignvariableop_86_border_off_kernelIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:ћ
AssignVariableOp_87AssignVariableOp#assignvariableop_87_border_off_biasIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:ќ
AssignVariableOp_88AssignVariableOp%assignvariableop_88_value_head_kernelIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:ћ
AssignVariableOp_89AssignVariableOp#assignvariableop_89_value_head_biasIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_90AssignVariableOpassignvariableop_90_totalIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_91AssignVariableOpassignvariableop_91_countIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 и
Identity_92Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_93IdentityIdentity_92:output:0^NoOp_1*
T0*
_output_shapes
: ц
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91*"
_acd_function_control_output(*
_output_shapes
 "#
identity_93Identity_93:output:0*¤
_input_shapesй
║: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_91:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ш
б
-__inference_border_off_layer_call_fn_10433916

inputs!
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_border_off_layer_call_and_return_conditional_losses_10430019w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
т
]
1__inference_concatenate_19_layer_call_fn_10432645
inputs_0
inputs_1
identity¤
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_19_layer_call_and_return_conditional_losses_10429013h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
§
д
1__inference_contract_7_3x3_layer_call_fn_10433006

inputs!
unknown: 
	unknown_0:
identityѕбStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_7_3x3_layer_call_and_return_conditional_losses_10429298w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Їё
█.
M__inference_gomoku_resnet_1_layer_call_and_return_conditional_losses_10430061

inputs2
expand_1_11x11_10428950:ђ&
expand_1_11x11_10428952:	ђ6
heuristic_detector_10428967:│*
heuristic_detector_10428969:	│6
heuristic_priority_10428984:│)
heuristic_priority_10428986:2
contract_1_5x5_10429001:ђ%
contract_1_5x5_10429003:/
expand_2_5x5_10429027:	 #
expand_2_5x5_10429029: 1
contract_2_3x3_10429044: %
contract_2_3x3_10429046:/
expand_3_5x5_10429078:	 #
expand_3_5x5_10429080: 1
contract_3_3x3_10429095: %
contract_3_3x3_10429097:/
expand_4_5x5_10429129:	 #
expand_4_5x5_10429131: 1
contract_4_3x3_10429146: %
contract_4_3x3_10429148:/
expand_5_5x5_10429180:	 #
expand_5_5x5_10429182: 1
contract_5_3x3_10429197: %
contract_5_3x3_10429199:/
expand_6_5x5_10429231:	 #
expand_6_5x5_10429233: 1
contract_6_3x3_10429248: %
contract_6_3x3_10429250:/
expand_7_5x5_10429282:	 #
expand_7_5x5_10429284: 1
contract_7_3x3_10429299: %
contract_7_3x3_10429301:/
expand_8_5x5_10429333:	 #
expand_8_5x5_10429335: 1
contract_8_3x3_10429350: %
contract_8_3x3_10429352:/
expand_9_5x5_10429384:	 #
expand_9_5x5_10429386: 1
contract_9_3x3_10429401: %
contract_9_3x3_10429403:0
expand_10_5x5_10429435:	 $
expand_10_5x5_10429437: 2
contract_10_3x3_10429452: &
contract_10_3x3_10429454:0
expand_11_5x5_10429486:	 $
expand_11_5x5_10429488: 2
contract_11_3x3_10429503: &
contract_11_3x3_10429505:0
expand_12_5x5_10429537:	 $
expand_12_5x5_10429539: 2
contract_12_3x3_10429554: &
contract_12_3x3_10429556:0
expand_13_5x5_10429588:	 $
expand_13_5x5_10429590: 2
contract_13_3x3_10429605: &
contract_13_3x3_10429607:0
expand_14_5x5_10429639:	 $
expand_14_5x5_10429641: 2
contract_14_3x3_10429656: &
contract_14_3x3_10429658:0
expand_15_5x5_10429690:	 $
expand_15_5x5_10429692: 2
contract_15_3x3_10429707: &
contract_15_3x3_10429709:0
expand_16_5x5_10429741:	 $
expand_16_5x5_10429743: 2
contract_16_3x3_10429758: &
contract_16_3x3_10429760:0
expand_17_5x5_10429792:	 $
expand_17_5x5_10429794: 2
contract_17_3x3_10429809: &
contract_17_3x3_10429811:0
expand_18_5x5_10429843:	 $
expand_18_5x5_10429845: 2
contract_18_3x3_10429860: &
contract_18_3x3_10429862:0
expand_19_5x5_10429894:	 $
expand_19_5x5_10429896: 2
contract_19_3x3_10429911: &
contract_19_3x3_10429913:0
expand_20_5x5_10429945:	 $
expand_20_5x5_10429947: 2
contract_20_3x3_10429962: &
contract_20_3x3_10429964:4
policy_aggregator_10429996:(
policy_aggregator_10429998:-
border_off_10430020:!
border_off_10430022:&
value_head_10430047:	т,!
value_head_10430049:
identity

identity_1ѕб"border_off/StatefulPartitionedCallб'contract_10_3x3/StatefulPartitionedCallб'contract_11_3x3/StatefulPartitionedCallб'contract_12_3x3/StatefulPartitionedCallб'contract_13_3x3/StatefulPartitionedCallб'contract_14_3x3/StatefulPartitionedCallб'contract_15_3x3/StatefulPartitionedCallб'contract_16_3x3/StatefulPartitionedCallб'contract_17_3x3/StatefulPartitionedCallб'contract_18_3x3/StatefulPartitionedCallб'contract_19_3x3/StatefulPartitionedCallб&contract_1_5x5/StatefulPartitionedCallб'contract_20_3x3/StatefulPartitionedCallб&contract_2_3x3/StatefulPartitionedCallб&contract_3_3x3/StatefulPartitionedCallб&contract_4_3x3/StatefulPartitionedCallб&contract_5_3x3/StatefulPartitionedCallб&contract_6_3x3/StatefulPartitionedCallб&contract_7_3x3/StatefulPartitionedCallб&contract_8_3x3/StatefulPartitionedCallб&contract_9_3x3/StatefulPartitionedCallб%expand_10_5x5/StatefulPartitionedCallб%expand_11_5x5/StatefulPartitionedCallб%expand_12_5x5/StatefulPartitionedCallб%expand_13_5x5/StatefulPartitionedCallб%expand_14_5x5/StatefulPartitionedCallб%expand_15_5x5/StatefulPartitionedCallб%expand_16_5x5/StatefulPartitionedCallб%expand_17_5x5/StatefulPartitionedCallб%expand_18_5x5/StatefulPartitionedCallб%expand_19_5x5/StatefulPartitionedCallб&expand_1_11x11/StatefulPartitionedCallб%expand_20_5x5/StatefulPartitionedCallб$expand_2_5x5/StatefulPartitionedCallб$expand_3_5x5/StatefulPartitionedCallб$expand_4_5x5/StatefulPartitionedCallб$expand_5_5x5/StatefulPartitionedCallб$expand_6_5x5/StatefulPartitionedCallб$expand_7_5x5/StatefulPartitionedCallб$expand_8_5x5/StatefulPartitionedCallб$expand_9_5x5/StatefulPartitionedCallб*heuristic_detector/StatefulPartitionedCallб*heuristic_priority/StatefulPartitionedCallб)policy_aggregator/StatefulPartitionedCallб"value_head/StatefulPartitionedCallџ
&expand_1_11x11/StatefulPartitionedCallStatefulPartitionedCallinputsexpand_1_11x11_10428950expand_1_11x11_10428952*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_expand_1_11x11_layer_call_and_return_conditional_losses_10428949ф
*heuristic_detector/StatefulPartitionedCallStatefulPartitionedCallinputsheuristic_detector_10428967heuristic_detector_10428969*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         │*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_heuristic_detector_layer_call_and_return_conditional_losses_10428966о
*heuristic_priority/StatefulPartitionedCallStatefulPartitionedCall3heuristic_detector/StatefulPartitionedCall:output:0heuristic_priority_10428984heuristic_priority_10428986*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_heuristic_priority_layer_call_and_return_conditional_losses_10428983┬
&contract_1_5x5/StatefulPartitionedCallStatefulPartitionedCall/expand_1_11x11/StatefulPartitionedCall:output:0contract_1_5x5_10429001contract_1_5x5_10429003*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_1_5x5_layer_call_and_return_conditional_losses_10429000░
concatenate_19/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0/contract_1_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_19_layer_call_and_return_conditional_losses_10429013▓
$expand_2_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_19/PartitionedCall:output:0expand_2_5x5_10429027expand_2_5x5_10429029*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_2_5x5_layer_call_and_return_conditional_losses_10429026└
&contract_2_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_2_5x5/StatefulPartitionedCall:output:0contract_2_3x3_10429044contract_2_3x3_10429046*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_2_3x3_layer_call_and_return_conditional_losses_10429043ю
skip_2/PartitionedCallPartitionedCall/contract_2_3x3/StatefulPartitionedCall:output:0/contract_1_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_2_layer_call_and_return_conditional_losses_10429055а
concatenate_20/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_20_layer_call_and_return_conditional_losses_10429064▓
$expand_3_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_20/PartitionedCall:output:0expand_3_5x5_10429078expand_3_5x5_10429080*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_3_5x5_layer_call_and_return_conditional_losses_10429077└
&contract_3_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_3_5x5/StatefulPartitionedCall:output:0contract_3_3x3_10429095contract_3_3x3_10429097*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_3_3x3_layer_call_and_return_conditional_losses_10429094ї
skip_3/PartitionedCallPartitionedCall/contract_3_3x3/StatefulPartitionedCall:output:0skip_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_3_layer_call_and_return_conditional_losses_10429106а
concatenate_21/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_21_layer_call_and_return_conditional_losses_10429115▓
$expand_4_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_21/PartitionedCall:output:0expand_4_5x5_10429129expand_4_5x5_10429131*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_4_5x5_layer_call_and_return_conditional_losses_10429128└
&contract_4_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_4_5x5/StatefulPartitionedCall:output:0contract_4_3x3_10429146contract_4_3x3_10429148*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_4_3x3_layer_call_and_return_conditional_losses_10429145ї
skip_4/PartitionedCallPartitionedCall/contract_4_3x3/StatefulPartitionedCall:output:0skip_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_4_layer_call_and_return_conditional_losses_10429157а
concatenate_22/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_22_layer_call_and_return_conditional_losses_10429166▓
$expand_5_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_22/PartitionedCall:output:0expand_5_5x5_10429180expand_5_5x5_10429182*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_5_5x5_layer_call_and_return_conditional_losses_10429179└
&contract_5_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_5_5x5/StatefulPartitionedCall:output:0contract_5_3x3_10429197contract_5_3x3_10429199*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_5_3x3_layer_call_and_return_conditional_losses_10429196ї
skip_5/PartitionedCallPartitionedCall/contract_5_3x3/StatefulPartitionedCall:output:0skip_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_5_layer_call_and_return_conditional_losses_10429208а
concatenate_23/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_23_layer_call_and_return_conditional_losses_10429217▓
$expand_6_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_23/PartitionedCall:output:0expand_6_5x5_10429231expand_6_5x5_10429233*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_6_5x5_layer_call_and_return_conditional_losses_10429230└
&contract_6_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_6_5x5/StatefulPartitionedCall:output:0contract_6_3x3_10429248contract_6_3x3_10429250*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_6_3x3_layer_call_and_return_conditional_losses_10429247ї
skip_6/PartitionedCallPartitionedCall/contract_6_3x3/StatefulPartitionedCall:output:0skip_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_6_layer_call_and_return_conditional_losses_10429259а
concatenate_24/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_24_layer_call_and_return_conditional_losses_10429268▓
$expand_7_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_24/PartitionedCall:output:0expand_7_5x5_10429282expand_7_5x5_10429284*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_7_5x5_layer_call_and_return_conditional_losses_10429281└
&contract_7_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_7_5x5/StatefulPartitionedCall:output:0contract_7_3x3_10429299contract_7_3x3_10429301*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_7_3x3_layer_call_and_return_conditional_losses_10429298ї
skip_7/PartitionedCallPartitionedCall/contract_7_3x3/StatefulPartitionedCall:output:0skip_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_7_layer_call_and_return_conditional_losses_10429310а
concatenate_25/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_25_layer_call_and_return_conditional_losses_10429319▓
$expand_8_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_25/PartitionedCall:output:0expand_8_5x5_10429333expand_8_5x5_10429335*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_8_5x5_layer_call_and_return_conditional_losses_10429332└
&contract_8_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_8_5x5/StatefulPartitionedCall:output:0contract_8_3x3_10429350contract_8_3x3_10429352*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_8_3x3_layer_call_and_return_conditional_losses_10429349ї
skip_8/PartitionedCallPartitionedCall/contract_8_3x3/StatefulPartitionedCall:output:0skip_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_8_layer_call_and_return_conditional_losses_10429361а
concatenate_26/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_26_layer_call_and_return_conditional_losses_10429370▓
$expand_9_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_26/PartitionedCall:output:0expand_9_5x5_10429384expand_9_5x5_10429386*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_9_5x5_layer_call_and_return_conditional_losses_10429383└
&contract_9_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_9_5x5/StatefulPartitionedCall:output:0contract_9_3x3_10429401contract_9_3x3_10429403*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_9_3x3_layer_call_and_return_conditional_losses_10429400ї
skip_9/PartitionedCallPartitionedCall/contract_9_3x3/StatefulPartitionedCall:output:0skip_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_9_layer_call_and_return_conditional_losses_10429412а
concatenate_27/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_27_layer_call_and_return_conditional_losses_10429421Х
%expand_10_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_27/PartitionedCall:output:0expand_10_5x5_10429435expand_10_5x5_10429437*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_10_5x5_layer_call_and_return_conditional_losses_10429434┼
'contract_10_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_10_5x5/StatefulPartitionedCall:output:0contract_10_3x3_10429452contract_10_3x3_10429454*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_10_3x3_layer_call_and_return_conditional_losses_10429451Ј
skip_10/PartitionedCallPartitionedCall0contract_10_3x3/StatefulPartitionedCall:output:0skip_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_10_layer_call_and_return_conditional_losses_10429463А
concatenate_28/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_28_layer_call_and_return_conditional_losses_10429472Х
%expand_11_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_28/PartitionedCall:output:0expand_11_5x5_10429486expand_11_5x5_10429488*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_11_5x5_layer_call_and_return_conditional_losses_10429485┼
'contract_11_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_11_5x5/StatefulPartitionedCall:output:0contract_11_3x3_10429503contract_11_3x3_10429505*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_11_3x3_layer_call_and_return_conditional_losses_10429502љ
skip_11/PartitionedCallPartitionedCall0contract_11_3x3/StatefulPartitionedCall:output:0 skip_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_11_layer_call_and_return_conditional_losses_10429514А
concatenate_29/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_29_layer_call_and_return_conditional_losses_10429523Х
%expand_12_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_29/PartitionedCall:output:0expand_12_5x5_10429537expand_12_5x5_10429539*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_12_5x5_layer_call_and_return_conditional_losses_10429536┼
'contract_12_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_12_5x5/StatefulPartitionedCall:output:0contract_12_3x3_10429554contract_12_3x3_10429556*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_12_3x3_layer_call_and_return_conditional_losses_10429553љ
skip_12/PartitionedCallPartitionedCall0contract_12_3x3/StatefulPartitionedCall:output:0 skip_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_12_layer_call_and_return_conditional_losses_10429565А
concatenate_30/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_30_layer_call_and_return_conditional_losses_10429574Х
%expand_13_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_30/PartitionedCall:output:0expand_13_5x5_10429588expand_13_5x5_10429590*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_13_5x5_layer_call_and_return_conditional_losses_10429587┼
'contract_13_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_13_5x5/StatefulPartitionedCall:output:0contract_13_3x3_10429605contract_13_3x3_10429607*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_13_3x3_layer_call_and_return_conditional_losses_10429604љ
skip_13/PartitionedCallPartitionedCall0contract_13_3x3/StatefulPartitionedCall:output:0 skip_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_13_layer_call_and_return_conditional_losses_10429616А
concatenate_31/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_31_layer_call_and_return_conditional_losses_10429625Х
%expand_14_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_31/PartitionedCall:output:0expand_14_5x5_10429639expand_14_5x5_10429641*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_14_5x5_layer_call_and_return_conditional_losses_10429638┼
'contract_14_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_14_5x5/StatefulPartitionedCall:output:0contract_14_3x3_10429656contract_14_3x3_10429658*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_14_3x3_layer_call_and_return_conditional_losses_10429655љ
skip_14/PartitionedCallPartitionedCall0contract_14_3x3/StatefulPartitionedCall:output:0 skip_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_14_layer_call_and_return_conditional_losses_10429667А
concatenate_32/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_14/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_32_layer_call_and_return_conditional_losses_10429676Х
%expand_15_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_32/PartitionedCall:output:0expand_15_5x5_10429690expand_15_5x5_10429692*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_15_5x5_layer_call_and_return_conditional_losses_10429689┼
'contract_15_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_15_5x5/StatefulPartitionedCall:output:0contract_15_3x3_10429707contract_15_3x3_10429709*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_15_3x3_layer_call_and_return_conditional_losses_10429706љ
skip_15/PartitionedCallPartitionedCall0contract_15_3x3/StatefulPartitionedCall:output:0 skip_14/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_15_layer_call_and_return_conditional_losses_10429718А
concatenate_33/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_33_layer_call_and_return_conditional_losses_10429727Х
%expand_16_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_33/PartitionedCall:output:0expand_16_5x5_10429741expand_16_5x5_10429743*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_16_5x5_layer_call_and_return_conditional_losses_10429740┼
'contract_16_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_16_5x5/StatefulPartitionedCall:output:0contract_16_3x3_10429758contract_16_3x3_10429760*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_16_3x3_layer_call_and_return_conditional_losses_10429757љ
skip_16/PartitionedCallPartitionedCall0contract_16_3x3/StatefulPartitionedCall:output:0 skip_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_16_layer_call_and_return_conditional_losses_10429769А
concatenate_34/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_34_layer_call_and_return_conditional_losses_10429778Х
%expand_17_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_34/PartitionedCall:output:0expand_17_5x5_10429792expand_17_5x5_10429794*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_17_5x5_layer_call_and_return_conditional_losses_10429791┼
'contract_17_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_17_5x5/StatefulPartitionedCall:output:0contract_17_3x3_10429809contract_17_3x3_10429811*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_17_3x3_layer_call_and_return_conditional_losses_10429808љ
skip_17/PartitionedCallPartitionedCall0contract_17_3x3/StatefulPartitionedCall:output:0 skip_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_17_layer_call_and_return_conditional_losses_10429820А
concatenate_35/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_35_layer_call_and_return_conditional_losses_10429829Х
%expand_18_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_35/PartitionedCall:output:0expand_18_5x5_10429843expand_18_5x5_10429845*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_18_5x5_layer_call_and_return_conditional_losses_10429842┼
'contract_18_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_18_5x5/StatefulPartitionedCall:output:0contract_18_3x3_10429860contract_18_3x3_10429862*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_18_3x3_layer_call_and_return_conditional_losses_10429859љ
skip_18/PartitionedCallPartitionedCall0contract_18_3x3/StatefulPartitionedCall:output:0 skip_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_18_layer_call_and_return_conditional_losses_10429871А
concatenate_36/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_18/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_36_layer_call_and_return_conditional_losses_10429880Х
%expand_19_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_36/PartitionedCall:output:0expand_19_5x5_10429894expand_19_5x5_10429896*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_19_5x5_layer_call_and_return_conditional_losses_10429893┼
'contract_19_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_19_5x5/StatefulPartitionedCall:output:0contract_19_3x3_10429911contract_19_3x3_10429913*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_19_3x3_layer_call_and_return_conditional_losses_10429910љ
skip_19/PartitionedCallPartitionedCall0contract_19_3x3/StatefulPartitionedCall:output:0 skip_18/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_19_layer_call_and_return_conditional_losses_10429922А
concatenate_37/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_19/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_37_layer_call_and_return_conditional_losses_10429931Х
%expand_20_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_37/PartitionedCall:output:0expand_20_5x5_10429945expand_20_5x5_10429947*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_20_5x5_layer_call_and_return_conditional_losses_10429944┼
'contract_20_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_20_5x5/StatefulPartitionedCall:output:0contract_20_3x3_10429962contract_20_3x3_10429964*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_20_3x3_layer_call_and_return_conditional_losses_10429961љ
skip_20/PartitionedCallPartitionedCall0contract_20_3x3/StatefulPartitionedCall:output:0 skip_19/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_20_layer_call_and_return_conditional_losses_10429973Д
all_value_input/PartitionedCallPartitionedCall0contract_20_3x3/StatefulPartitionedCall:output:0'concatenate_19/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_all_value_input_layer_call_and_return_conditional_losses_10429982┐
)policy_aggregator/StatefulPartitionedCallStatefulPartitionedCall skip_20/PartitionedCall:output:0policy_aggregator_10429996policy_aggregator_10429998*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_policy_aggregator_layer_call_and_return_conditional_losses_10429995­
 flat_value_input/PartitionedCallPartitionedCall(all_value_input/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         т,* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_flat_value_input_layer_call_and_return_conditional_losses_10430007х
"border_off/StatefulPartitionedCallStatefulPartitionedCall2policy_aggregator/StatefulPartitionedCall:output:0border_off_10430020border_off_10430022*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_border_off_layer_call_and_return_conditional_losses_10430019`
tf.math.truediv_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ╚Bе
tf.math.truediv_1/truedivRealDiv)flat_value_input/PartitionedCall:output:0$tf.math.truediv_1/truediv/y:output:0*
T0*(
_output_shapes
:         т,ж
flat_logits/PartitionedCallPartitionedCall+border_off/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ж* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_flat_logits_layer_call_and_return_conditional_losses_10430033ў
"value_head/StatefulPartitionedCallStatefulPartitionedCalltf.math.truediv_1/truediv:z:0value_head_10430047value_head_10430049*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_value_head_layer_call_and_return_conditional_losses_10430046Р
policy_head/PartitionedCallPartitionedCall$flat_logits/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ж* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_policy_head_layer_call_and_return_conditional_losses_10430057t
IdentityIdentity$policy_head/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ж|

Identity_1Identity+value_head/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Ь
NoOpNoOp#^border_off/StatefulPartitionedCall(^contract_10_3x3/StatefulPartitionedCall(^contract_11_3x3/StatefulPartitionedCall(^contract_12_3x3/StatefulPartitionedCall(^contract_13_3x3/StatefulPartitionedCall(^contract_14_3x3/StatefulPartitionedCall(^contract_15_3x3/StatefulPartitionedCall(^contract_16_3x3/StatefulPartitionedCall(^contract_17_3x3/StatefulPartitionedCall(^contract_18_3x3/StatefulPartitionedCall(^contract_19_3x3/StatefulPartitionedCall'^contract_1_5x5/StatefulPartitionedCall(^contract_20_3x3/StatefulPartitionedCall'^contract_2_3x3/StatefulPartitionedCall'^contract_3_3x3/StatefulPartitionedCall'^contract_4_3x3/StatefulPartitionedCall'^contract_5_3x3/StatefulPartitionedCall'^contract_6_3x3/StatefulPartitionedCall'^contract_7_3x3/StatefulPartitionedCall'^contract_8_3x3/StatefulPartitionedCall'^contract_9_3x3/StatefulPartitionedCall&^expand_10_5x5/StatefulPartitionedCall&^expand_11_5x5/StatefulPartitionedCall&^expand_12_5x5/StatefulPartitionedCall&^expand_13_5x5/StatefulPartitionedCall&^expand_14_5x5/StatefulPartitionedCall&^expand_15_5x5/StatefulPartitionedCall&^expand_16_5x5/StatefulPartitionedCall&^expand_17_5x5/StatefulPartitionedCall&^expand_18_5x5/StatefulPartitionedCall&^expand_19_5x5/StatefulPartitionedCall'^expand_1_11x11/StatefulPartitionedCall&^expand_20_5x5/StatefulPartitionedCall%^expand_2_5x5/StatefulPartitionedCall%^expand_3_5x5/StatefulPartitionedCall%^expand_4_5x5/StatefulPartitionedCall%^expand_5_5x5/StatefulPartitionedCall%^expand_6_5x5/StatefulPartitionedCall%^expand_7_5x5/StatefulPartitionedCall%^expand_8_5x5/StatefulPartitionedCall%^expand_9_5x5/StatefulPartitionedCall+^heuristic_detector/StatefulPartitionedCall+^heuristic_priority/StatefulPartitionedCall*^policy_aggregator/StatefulPartitionedCall#^value_head/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*С
_input_shapesм
¤:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"border_off/StatefulPartitionedCall"border_off/StatefulPartitionedCall2R
'contract_10_3x3/StatefulPartitionedCall'contract_10_3x3/StatefulPartitionedCall2R
'contract_11_3x3/StatefulPartitionedCall'contract_11_3x3/StatefulPartitionedCall2R
'contract_12_3x3/StatefulPartitionedCall'contract_12_3x3/StatefulPartitionedCall2R
'contract_13_3x3/StatefulPartitionedCall'contract_13_3x3/StatefulPartitionedCall2R
'contract_14_3x3/StatefulPartitionedCall'contract_14_3x3/StatefulPartitionedCall2R
'contract_15_3x3/StatefulPartitionedCall'contract_15_3x3/StatefulPartitionedCall2R
'contract_16_3x3/StatefulPartitionedCall'contract_16_3x3/StatefulPartitionedCall2R
'contract_17_3x3/StatefulPartitionedCall'contract_17_3x3/StatefulPartitionedCall2R
'contract_18_3x3/StatefulPartitionedCall'contract_18_3x3/StatefulPartitionedCall2R
'contract_19_3x3/StatefulPartitionedCall'contract_19_3x3/StatefulPartitionedCall2P
&contract_1_5x5/StatefulPartitionedCall&contract_1_5x5/StatefulPartitionedCall2R
'contract_20_3x3/StatefulPartitionedCall'contract_20_3x3/StatefulPartitionedCall2P
&contract_2_3x3/StatefulPartitionedCall&contract_2_3x3/StatefulPartitionedCall2P
&contract_3_3x3/StatefulPartitionedCall&contract_3_3x3/StatefulPartitionedCall2P
&contract_4_3x3/StatefulPartitionedCall&contract_4_3x3/StatefulPartitionedCall2P
&contract_5_3x3/StatefulPartitionedCall&contract_5_3x3/StatefulPartitionedCall2P
&contract_6_3x3/StatefulPartitionedCall&contract_6_3x3/StatefulPartitionedCall2P
&contract_7_3x3/StatefulPartitionedCall&contract_7_3x3/StatefulPartitionedCall2P
&contract_8_3x3/StatefulPartitionedCall&contract_8_3x3/StatefulPartitionedCall2P
&contract_9_3x3/StatefulPartitionedCall&contract_9_3x3/StatefulPartitionedCall2N
%expand_10_5x5/StatefulPartitionedCall%expand_10_5x5/StatefulPartitionedCall2N
%expand_11_5x5/StatefulPartitionedCall%expand_11_5x5/StatefulPartitionedCall2N
%expand_12_5x5/StatefulPartitionedCall%expand_12_5x5/StatefulPartitionedCall2N
%expand_13_5x5/StatefulPartitionedCall%expand_13_5x5/StatefulPartitionedCall2N
%expand_14_5x5/StatefulPartitionedCall%expand_14_5x5/StatefulPartitionedCall2N
%expand_15_5x5/StatefulPartitionedCall%expand_15_5x5/StatefulPartitionedCall2N
%expand_16_5x5/StatefulPartitionedCall%expand_16_5x5/StatefulPartitionedCall2N
%expand_17_5x5/StatefulPartitionedCall%expand_17_5x5/StatefulPartitionedCall2N
%expand_18_5x5/StatefulPartitionedCall%expand_18_5x5/StatefulPartitionedCall2N
%expand_19_5x5/StatefulPartitionedCall%expand_19_5x5/StatefulPartitionedCall2P
&expand_1_11x11/StatefulPartitionedCall&expand_1_11x11/StatefulPartitionedCall2N
%expand_20_5x5/StatefulPartitionedCall%expand_20_5x5/StatefulPartitionedCall2L
$expand_2_5x5/StatefulPartitionedCall$expand_2_5x5/StatefulPartitionedCall2L
$expand_3_5x5/StatefulPartitionedCall$expand_3_5x5/StatefulPartitionedCall2L
$expand_4_5x5/StatefulPartitionedCall$expand_4_5x5/StatefulPartitionedCall2L
$expand_5_5x5/StatefulPartitionedCall$expand_5_5x5/StatefulPartitionedCall2L
$expand_6_5x5/StatefulPartitionedCall$expand_6_5x5/StatefulPartitionedCall2L
$expand_7_5x5/StatefulPartitionedCall$expand_7_5x5/StatefulPartitionedCall2L
$expand_8_5x5/StatefulPartitionedCall$expand_8_5x5/StatefulPartitionedCall2L
$expand_9_5x5/StatefulPartitionedCall$expand_9_5x5/StatefulPartitionedCall2X
*heuristic_detector/StatefulPartitionedCall*heuristic_detector/StatefulPartitionedCall2X
*heuristic_priority/StatefulPartitionedCall*heuristic_priority/StatefulPartitionedCall2V
)policy_aggregator/StatefulPartitionedCall)policy_aggregator/StatefulPartitionedCall2H
"value_head/StatefulPartitionedCall"value_head/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ќ
ё
K__inference_expand_17_5x5_layer_call_and_return_conditional_losses_10433647

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:          m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
ў
є
M__inference_contract_14_3x3_layer_call_and_return_conditional_losses_10429655

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
ж
n
D__inference_skip_4_layer_call_and_return_conditional_losses_10429157

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
Ћ
Ѓ
J__inference_expand_3_5x5_layer_call_and_return_conditional_losses_10429077

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:          m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
Ћ
Ѓ
J__inference_expand_6_5x5_layer_call_and_return_conditional_losses_10429230

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:          m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
Ћ
Ѓ
J__inference_expand_8_5x5_layer_call_and_return_conditional_losses_10433062

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:          m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
Ы
q
E__inference_skip_19_layer_call_and_return_conditional_losses_10433809
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
ш
x
L__inference_concatenate_29_layer_call_and_return_conditional_losses_10433302
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
ќ
ё
K__inference_expand_20_5x5_layer_call_and_return_conditional_losses_10433842

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:          m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
ш
x
L__inference_concatenate_26_layer_call_and_return_conditional_losses_10433107
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
ќ
ё
K__inference_expand_12_5x5_layer_call_and_return_conditional_losses_10433322

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:          m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
ш
x
L__inference_concatenate_24_layer_call_and_return_conditional_losses_10432977
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
щ
ц
/__inference_expand_9_5x5_layer_call_fn_10433116

inputs!
unknown:	 
	unknown_0: 
identityѕбStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_9_5x5_layer_call_and_return_conditional_losses_10429383w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
і
і
P__inference_heuristic_priority_layer_call_and_return_conditional_losses_10432619

inputs9
conv2d_readvariableop_resource:│-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:│*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         X
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:         _
IdentityIdentityTanh:y:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         │: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         │
 
_user_specified_nameinputs
ы
p
D__inference_skip_5_layer_call_and_return_conditional_losses_10432899
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
Ќ
Ё
L__inference_contract_5_3x3_layer_call_and_return_conditional_losses_10432887

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Ќ
Ё
L__inference_contract_4_3x3_layer_call_and_return_conditional_losses_10432822

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
ы
p
D__inference_skip_9_layer_call_and_return_conditional_losses_10433159
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
ь
v
L__inference_concatenate_32_layer_call_and_return_conditional_losses_10429676

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :}
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
ў
є
M__inference_contract_15_3x3_layer_call_and_return_conditional_losses_10433537

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
О
V
*__inference_skip_20_layer_call_fn_10433868
inputs_0
inputs_1
identity╚
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_20_layer_call_and_return_conditional_losses_10429973h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
к
O
3__inference_flat_value_input_layer_call_fn_10433931

inputs
identityй
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         т,* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_flat_value_input_layer_call_and_return_conditional_losses_10430007a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         т,"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Ќ
Ё
L__inference_contract_8_3x3_layer_call_and_return_conditional_losses_10429349

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Ќ
Ё
L__inference_contract_6_3x3_layer_call_and_return_conditional_losses_10432952

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Н
U
)__inference_skip_9_layer_call_fn_10433153
inputs_0
inputs_1
identityК
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_9_layer_call_and_return_conditional_losses_10429412h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
Ъ
Є
L__inference_expand_1_11x11_layer_call_and_return_conditional_losses_10432599

inputs9
conv2d_readvariableop_resource:ђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:ђ*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђa
SoftplusSoftplusBiasAdd:output:0*
T0*0
_output_shapes
:         ђn
IdentityIdentitySoftplus:activations:0^NoOp*
T0*0
_output_shapes
:         ђw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Ќ
Ё
L__inference_contract_3_3x3_layer_call_and_return_conditional_losses_10429094

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
ў
є
M__inference_contract_12_3x3_layer_call_and_return_conditional_losses_10429553

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
§
д
1__inference_contract_3_3x3_layer_call_fn_10432746

inputs!
unknown: 
	unknown_0:
identityѕбStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_3_3x3_layer_call_and_return_conditional_losses_10429094w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Н
U
)__inference_skip_2_layer_call_fn_10432698
inputs_0
inputs_1
identityК
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_2_layer_call_and_return_conditional_losses_10429055h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
О
V
*__inference_skip_13_layer_call_fn_10433413
inputs_0
inputs_1
identity╚
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_13_layer_call_and_return_conditional_losses_10429616h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
Ж
o
E__inference_skip_10_layer_call_and_return_conditional_losses_10429463

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
ў
є
M__inference_contract_13_3x3_layer_call_and_return_conditional_losses_10433407

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
ь
v
L__inference_concatenate_28_layer_call_and_return_conditional_losses_10429472

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :}
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
т
]
1__inference_concatenate_33_layer_call_fn_10433555
inputs_0
inputs_1
identity¤
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_33_layer_call_and_return_conditional_losses_10429727h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
О
V
*__inference_skip_17_layer_call_fn_10433673
inputs_0
inputs_1
identity╚
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_17_layer_call_and_return_conditional_losses_10429820h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
Ж
o
E__inference_skip_11_layer_call_and_return_conditional_losses_10429514

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
Ъ
Є
L__inference_expand_1_11x11_layer_call_and_return_conditional_losses_10428949

inputs9
conv2d_readvariableop_resource:ђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:ђ*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђa
SoftplusSoftplusBiasAdd:output:0*
T0*0
_output_shapes
:         ђn
IdentityIdentitySoftplus:activations:0^NoOp*
T0*0
_output_shapes
:         ђw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ќ
ё
K__inference_expand_20_5x5_layer_call_and_return_conditional_losses_10429944

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:          m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
Ћ
Ѓ
J__inference_expand_6_5x5_layer_call_and_return_conditional_losses_10432932

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:          m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
ќ
ё
K__inference_expand_10_5x5_layer_call_and_return_conditional_losses_10433192

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:          m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
ш
x
L__inference_concatenate_25_layer_call_and_return_conditional_losses_10433042
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
ь
v
L__inference_concatenate_33_layer_call_and_return_conditional_losses_10429727

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :}
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
ќ
ё
K__inference_expand_18_5x5_layer_call_and_return_conditional_losses_10433712

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:          m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
Ы
q
E__inference_skip_14_layer_call_and_return_conditional_losses_10433484
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
О
V
*__inference_skip_12_layer_call_fn_10433348
inputs_0
inputs_1
identity╚
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_12_layer_call_and_return_conditional_losses_10429565h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
ь
v
L__inference_concatenate_37_layer_call_and_return_conditional_losses_10429931

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :}
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
т
]
1__inference_concatenate_24_layer_call_fn_10432970
inputs_0
inputs_1
identity¤
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_24_layer_call_and_return_conditional_losses_10429268h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
щ
ц
/__inference_expand_5_5x5_layer_call_fn_10432856

inputs!
unknown:	 
	unknown_0: 
identityѕбStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_5_5x5_layer_call_and_return_conditional_losses_10429179w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
 
Д
2__inference_contract_20_3x3_layer_call_fn_10433851

inputs!
unknown: 
	unknown_0:
identityѕбStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_20_3x3_layer_call_and_return_conditional_losses_10429961w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
ш
x
L__inference_concatenate_28_layer_call_and_return_conditional_losses_10433237
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
Ы
q
E__inference_skip_20_layer_call_and_return_conditional_losses_10433874
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
Ў

Щ
H__inference_value_head_layer_call_and_return_conditional_losses_10433978

inputs1
matmul_readvariableop_resource:	т,-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	т,*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:         W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         т,: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         т,
 
_user_specified_nameinputs
ќ
ё
K__inference_expand_19_5x5_layer_call_and_return_conditional_losses_10433777

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:          m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
ќ
ё
K__inference_expand_15_5x5_layer_call_and_return_conditional_losses_10429689

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:          m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
 
Д
2__inference_contract_13_3x3_layer_call_fn_10433396

inputs!
unknown: 
	unknown_0:
identityѕбStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_13_3x3_layer_call_and_return_conditional_losses_10429604w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
ў
є
M__inference_contract_13_3x3_layer_call_and_return_conditional_losses_10429604

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Ж
o
E__inference_skip_14_layer_call_and_return_conditional_losses_10429667

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
Ы
q
E__inference_skip_11_layer_call_and_return_conditional_losses_10433289
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
ж
n
D__inference_skip_8_layer_call_and_return_conditional_losses_10429361

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
ж
n
D__inference_skip_5_layer_call_and_return_conditional_losses_10429208

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
ў
є
M__inference_contract_18_3x3_layer_call_and_return_conditional_losses_10429859

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Ћ
Ѓ
J__inference_expand_2_5x5_layer_call_and_return_conditional_losses_10432672

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:          m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
Н
U
)__inference_skip_5_layer_call_fn_10432893
inputs_0
inputs_1
identityК
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_5_layer_call_and_return_conditional_losses_10429208h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
ч
Ц
0__inference_expand_13_5x5_layer_call_fn_10433376

inputs!
unknown:	 
	unknown_0: 
identityѕбStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_13_5x5_layer_call_and_return_conditional_losses_10429587w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
ь
v
L__inference_concatenate_31_layer_call_and_return_conditional_losses_10429625

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :}
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
ќ
ё
K__inference_expand_16_5x5_layer_call_and_return_conditional_losses_10429740

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:          m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
ў
є
M__inference_contract_14_3x3_layer_call_and_return_conditional_losses_10433472

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
ш
x
L__inference_concatenate_21_layer_call_and_return_conditional_losses_10432782
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
ч
Ц
0__inference_expand_10_5x5_layer_call_fn_10433181

inputs!
unknown:	 
	unknown_0: 
identityѕбStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_10_5x5_layer_call_and_return_conditional_losses_10429434w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
Н
U
)__inference_skip_6_layer_call_fn_10432958
inputs_0
inputs_1
identityК
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_6_layer_call_and_return_conditional_losses_10429259h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
ы
p
D__inference_skip_6_layer_call_and_return_conditional_losses_10432964
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
§
д
1__inference_contract_9_3x3_layer_call_fn_10433136

inputs!
unknown: 
	unknown_0:
identityѕбStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_9_3x3_layer_call_and_return_conditional_losses_10429400w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
о
e
I__inference_policy_head_layer_call_and_return_conditional_losses_10430057

inputs
identityM
SoftmaxSoftmaxinputs*
T0*(
_output_shapes
:         жZ
IdentityIdentitySoftmax:softmax:0*
T0*(
_output_shapes
:         ж"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ж:P L
(
_output_shapes
:         ж
 
_user_specified_nameinputs
ш
x
L__inference_concatenate_30_layer_call_and_return_conditional_losses_10433367
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
ў
є
M__inference_contract_17_3x3_layer_call_and_return_conditional_losses_10433667

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
ч
Ц
0__inference_expand_14_5x5_layer_call_fn_10433441

inputs!
unknown:	 
	unknown_0: 
identityѕбStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_14_5x5_layer_call_and_return_conditional_losses_10429638w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
Ж
o
E__inference_skip_18_layer_call_and_return_conditional_losses_10429871

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
Џ
є
L__inference_contract_1_5x5_layer_call_and_return_conditional_losses_10429000

inputs9
conv2d_readvariableop_resource:ђ-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:ђ*
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
ы
p
D__inference_skip_3_layer_call_and_return_conditional_losses_10432769
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
ч
Ц
0__inference_expand_17_5x5_layer_call_fn_10433636

inputs!
unknown:	 
	unknown_0: 
identityѕбStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_17_5x5_layer_call_and_return_conditional_losses_10429791w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
ќ
ё
K__inference_expand_13_5x5_layer_call_and_return_conditional_losses_10429587

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:          m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
г

Ђ
H__inference_border_off_layer_call_and_return_conditional_losses_10430019

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ж
n
D__inference_skip_2_layer_call_and_return_conditional_losses_10429055

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
ь
v
L__inference_concatenate_26_layer_call_and_return_conditional_losses_10429370

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :}
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
т
]
1__inference_concatenate_26_layer_call_fn_10433100
inputs_0
inputs_1
identity¤
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_26_layer_call_and_return_conditional_losses_10429370h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
Ќ
Ё
L__inference_contract_5_3x3_layer_call_and_return_conditional_losses_10429196

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Ќ
Ё
L__inference_contract_2_3x3_layer_call_and_return_conditional_losses_10432692

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Ќ
Ё
L__inference_contract_8_3x3_layer_call_and_return_conditional_losses_10433082

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
ў
є
M__inference_contract_16_3x3_layer_call_and_return_conditional_losses_10429757

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
ш
x
L__inference_concatenate_31_layer_call_and_return_conditional_losses_10433432
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
Ы
q
E__inference_skip_15_layer_call_and_return_conditional_losses_10433549
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
§
д
1__inference_contract_2_3x3_layer_call_fn_10432681

inputs!
unknown: 
	unknown_0:
identityѕбStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_2_3x3_layer_call_and_return_conditional_losses_10429043w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
щ
ц
/__inference_expand_7_5x5_layer_call_fn_10432986

inputs!
unknown:	 
	unknown_0: 
identityѕбStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_7_5x5_layer_call_and_return_conditional_losses_10429281w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
ж
n
D__inference_skip_7_layer_call_and_return_conditional_losses_10429310

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
ь
v
L__inference_concatenate_36_layer_call_and_return_conditional_losses_10429880

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :}
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
у
^
2__inference_all_value_input_layer_call_fn_10433900
inputs_0
inputs_1
identityл
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_all_value_input_layer_call_and_return_conditional_losses_10429982h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         	:Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         	
"
_user_specified_name
inputs/1
Ћ
Ѓ
J__inference_expand_8_5x5_layer_call_and_return_conditional_losses_10429332

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:          m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
ў
є
M__inference_contract_17_3x3_layer_call_and_return_conditional_losses_10429808

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
ь
v
L__inference_concatenate_19_layer_call_and_return_conditional_losses_10429013

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :}
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
Н
U
)__inference_skip_8_layer_call_fn_10433088
inputs_0
inputs_1
identityК
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_8_layer_call_and_return_conditional_losses_10429361h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
ш
x
L__inference_concatenate_34_layer_call_and_return_conditional_losses_10433627
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
Ћ
Ѓ
J__inference_expand_5_5x5_layer_call_and_return_conditional_losses_10429179

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:          m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
ў
є
M__inference_contract_19_3x3_layer_call_and_return_conditional_losses_10429910

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
ш
x
L__inference_concatenate_19_layer_call_and_return_conditional_losses_10432652
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
ш
x
L__inference_concatenate_22_layer_call_and_return_conditional_losses_10432847
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
ў
є
M__inference_contract_11_3x3_layer_call_and_return_conditional_losses_10433277

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
ќ
ё
K__inference_expand_14_5x5_layer_call_and_return_conditional_losses_10433452

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:          m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
ў
є
M__inference_contract_20_3x3_layer_call_and_return_conditional_losses_10429961

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
О
V
*__inference_skip_16_layer_call_fn_10433608
inputs_0
inputs_1
identity╚
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_16_layer_call_and_return_conditional_losses_10429769h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
ше
┬&
!__inference__traced_save_10434278
file_prefix8
4savev2_heuristic_detector_kernel_read_readvariableop6
2savev2_heuristic_detector_bias_read_readvariableop4
0savev2_expand_1_11x11_kernel_read_readvariableop2
.savev2_expand_1_11x11_bias_read_readvariableop8
4savev2_heuristic_priority_kernel_read_readvariableop6
2savev2_heuristic_priority_bias_read_readvariableop4
0savev2_contract_1_5x5_kernel_read_readvariableop2
.savev2_contract_1_5x5_bias_read_readvariableop2
.savev2_expand_2_5x5_kernel_read_readvariableop0
,savev2_expand_2_5x5_bias_read_readvariableop4
0savev2_contract_2_3x3_kernel_read_readvariableop2
.savev2_contract_2_3x3_bias_read_readvariableop2
.savev2_expand_3_5x5_kernel_read_readvariableop0
,savev2_expand_3_5x5_bias_read_readvariableop4
0savev2_contract_3_3x3_kernel_read_readvariableop2
.savev2_contract_3_3x3_bias_read_readvariableop2
.savev2_expand_4_5x5_kernel_read_readvariableop0
,savev2_expand_4_5x5_bias_read_readvariableop4
0savev2_contract_4_3x3_kernel_read_readvariableop2
.savev2_contract_4_3x3_bias_read_readvariableop2
.savev2_expand_5_5x5_kernel_read_readvariableop0
,savev2_expand_5_5x5_bias_read_readvariableop4
0savev2_contract_5_3x3_kernel_read_readvariableop2
.savev2_contract_5_3x3_bias_read_readvariableop2
.savev2_expand_6_5x5_kernel_read_readvariableop0
,savev2_expand_6_5x5_bias_read_readvariableop4
0savev2_contract_6_3x3_kernel_read_readvariableop2
.savev2_contract_6_3x3_bias_read_readvariableop2
.savev2_expand_7_5x5_kernel_read_readvariableop0
,savev2_expand_7_5x5_bias_read_readvariableop4
0savev2_contract_7_3x3_kernel_read_readvariableop2
.savev2_contract_7_3x3_bias_read_readvariableop2
.savev2_expand_8_5x5_kernel_read_readvariableop0
,savev2_expand_8_5x5_bias_read_readvariableop4
0savev2_contract_8_3x3_kernel_read_readvariableop2
.savev2_contract_8_3x3_bias_read_readvariableop2
.savev2_expand_9_5x5_kernel_read_readvariableop0
,savev2_expand_9_5x5_bias_read_readvariableop4
0savev2_contract_9_3x3_kernel_read_readvariableop2
.savev2_contract_9_3x3_bias_read_readvariableop3
/savev2_expand_10_5x5_kernel_read_readvariableop1
-savev2_expand_10_5x5_bias_read_readvariableop5
1savev2_contract_10_3x3_kernel_read_readvariableop3
/savev2_contract_10_3x3_bias_read_readvariableop3
/savev2_expand_11_5x5_kernel_read_readvariableop1
-savev2_expand_11_5x5_bias_read_readvariableop5
1savev2_contract_11_3x3_kernel_read_readvariableop3
/savev2_contract_11_3x3_bias_read_readvariableop3
/savev2_expand_12_5x5_kernel_read_readvariableop1
-savev2_expand_12_5x5_bias_read_readvariableop5
1savev2_contract_12_3x3_kernel_read_readvariableop3
/savev2_contract_12_3x3_bias_read_readvariableop3
/savev2_expand_13_5x5_kernel_read_readvariableop1
-savev2_expand_13_5x5_bias_read_readvariableop5
1savev2_contract_13_3x3_kernel_read_readvariableop3
/savev2_contract_13_3x3_bias_read_readvariableop3
/savev2_expand_14_5x5_kernel_read_readvariableop1
-savev2_expand_14_5x5_bias_read_readvariableop5
1savev2_contract_14_3x3_kernel_read_readvariableop3
/savev2_contract_14_3x3_bias_read_readvariableop3
/savev2_expand_15_5x5_kernel_read_readvariableop1
-savev2_expand_15_5x5_bias_read_readvariableop5
1savev2_contract_15_3x3_kernel_read_readvariableop3
/savev2_contract_15_3x3_bias_read_readvariableop3
/savev2_expand_16_5x5_kernel_read_readvariableop1
-savev2_expand_16_5x5_bias_read_readvariableop5
1savev2_contract_16_3x3_kernel_read_readvariableop3
/savev2_contract_16_3x3_bias_read_readvariableop3
/savev2_expand_17_5x5_kernel_read_readvariableop1
-savev2_expand_17_5x5_bias_read_readvariableop5
1savev2_contract_17_3x3_kernel_read_readvariableop3
/savev2_contract_17_3x3_bias_read_readvariableop3
/savev2_expand_18_5x5_kernel_read_readvariableop1
-savev2_expand_18_5x5_bias_read_readvariableop5
1savev2_contract_18_3x3_kernel_read_readvariableop3
/savev2_contract_18_3x3_bias_read_readvariableop3
/savev2_expand_19_5x5_kernel_read_readvariableop1
-savev2_expand_19_5x5_bias_read_readvariableop5
1savev2_contract_19_3x3_kernel_read_readvariableop3
/savev2_contract_19_3x3_bias_read_readvariableop3
/savev2_expand_20_5x5_kernel_read_readvariableop1
-savev2_expand_20_5x5_bias_read_readvariableop5
1savev2_contract_20_3x3_kernel_read_readvariableop3
/savev2_contract_20_3x3_bias_read_readvariableop7
3savev2_policy_aggregator_kernel_read_readvariableop5
1savev2_policy_aggregator_bias_read_readvariableop0
,savev2_border_off_kernel_read_readvariableop.
*savev2_border_off_bias_read_readvariableop0
,savev2_value_head_kernel_read_readvariableop.
*savev2_value_head_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1ѕбMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partЂ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : Њ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: њ)
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:]*
dtype0*╗(
value▒(B«(]B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-25/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-25/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-26/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-27/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-27/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-28/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-28/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-29/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-29/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-30/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-30/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-31/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-31/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-32/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-32/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-33/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-33/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-34/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-34/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-35/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-35/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-36/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-36/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-37/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-37/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-38/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-38/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-39/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-39/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-40/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-40/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-41/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-41/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-42/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-42/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-43/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-43/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-44/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-44/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHф
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:]*
dtype0*¤
value┼B┬]B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ­$
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_heuristic_detector_kernel_read_readvariableop2savev2_heuristic_detector_bias_read_readvariableop0savev2_expand_1_11x11_kernel_read_readvariableop.savev2_expand_1_11x11_bias_read_readvariableop4savev2_heuristic_priority_kernel_read_readvariableop2savev2_heuristic_priority_bias_read_readvariableop0savev2_contract_1_5x5_kernel_read_readvariableop.savev2_contract_1_5x5_bias_read_readvariableop.savev2_expand_2_5x5_kernel_read_readvariableop,savev2_expand_2_5x5_bias_read_readvariableop0savev2_contract_2_3x3_kernel_read_readvariableop.savev2_contract_2_3x3_bias_read_readvariableop.savev2_expand_3_5x5_kernel_read_readvariableop,savev2_expand_3_5x5_bias_read_readvariableop0savev2_contract_3_3x3_kernel_read_readvariableop.savev2_contract_3_3x3_bias_read_readvariableop.savev2_expand_4_5x5_kernel_read_readvariableop,savev2_expand_4_5x5_bias_read_readvariableop0savev2_contract_4_3x3_kernel_read_readvariableop.savev2_contract_4_3x3_bias_read_readvariableop.savev2_expand_5_5x5_kernel_read_readvariableop,savev2_expand_5_5x5_bias_read_readvariableop0savev2_contract_5_3x3_kernel_read_readvariableop.savev2_contract_5_3x3_bias_read_readvariableop.savev2_expand_6_5x5_kernel_read_readvariableop,savev2_expand_6_5x5_bias_read_readvariableop0savev2_contract_6_3x3_kernel_read_readvariableop.savev2_contract_6_3x3_bias_read_readvariableop.savev2_expand_7_5x5_kernel_read_readvariableop,savev2_expand_7_5x5_bias_read_readvariableop0savev2_contract_7_3x3_kernel_read_readvariableop.savev2_contract_7_3x3_bias_read_readvariableop.savev2_expand_8_5x5_kernel_read_readvariableop,savev2_expand_8_5x5_bias_read_readvariableop0savev2_contract_8_3x3_kernel_read_readvariableop.savev2_contract_8_3x3_bias_read_readvariableop.savev2_expand_9_5x5_kernel_read_readvariableop,savev2_expand_9_5x5_bias_read_readvariableop0savev2_contract_9_3x3_kernel_read_readvariableop.savev2_contract_9_3x3_bias_read_readvariableop/savev2_expand_10_5x5_kernel_read_readvariableop-savev2_expand_10_5x5_bias_read_readvariableop1savev2_contract_10_3x3_kernel_read_readvariableop/savev2_contract_10_3x3_bias_read_readvariableop/savev2_expand_11_5x5_kernel_read_readvariableop-savev2_expand_11_5x5_bias_read_readvariableop1savev2_contract_11_3x3_kernel_read_readvariableop/savev2_contract_11_3x3_bias_read_readvariableop/savev2_expand_12_5x5_kernel_read_readvariableop-savev2_expand_12_5x5_bias_read_readvariableop1savev2_contract_12_3x3_kernel_read_readvariableop/savev2_contract_12_3x3_bias_read_readvariableop/savev2_expand_13_5x5_kernel_read_readvariableop-savev2_expand_13_5x5_bias_read_readvariableop1savev2_contract_13_3x3_kernel_read_readvariableop/savev2_contract_13_3x3_bias_read_readvariableop/savev2_expand_14_5x5_kernel_read_readvariableop-savev2_expand_14_5x5_bias_read_readvariableop1savev2_contract_14_3x3_kernel_read_readvariableop/savev2_contract_14_3x3_bias_read_readvariableop/savev2_expand_15_5x5_kernel_read_readvariableop-savev2_expand_15_5x5_bias_read_readvariableop1savev2_contract_15_3x3_kernel_read_readvariableop/savev2_contract_15_3x3_bias_read_readvariableop/savev2_expand_16_5x5_kernel_read_readvariableop-savev2_expand_16_5x5_bias_read_readvariableop1savev2_contract_16_3x3_kernel_read_readvariableop/savev2_contract_16_3x3_bias_read_readvariableop/savev2_expand_17_5x5_kernel_read_readvariableop-savev2_expand_17_5x5_bias_read_readvariableop1savev2_contract_17_3x3_kernel_read_readvariableop/savev2_contract_17_3x3_bias_read_readvariableop/savev2_expand_18_5x5_kernel_read_readvariableop-savev2_expand_18_5x5_bias_read_readvariableop1savev2_contract_18_3x3_kernel_read_readvariableop/savev2_contract_18_3x3_bias_read_readvariableop/savev2_expand_19_5x5_kernel_read_readvariableop-savev2_expand_19_5x5_bias_read_readvariableop1savev2_contract_19_3x3_kernel_read_readvariableop/savev2_contract_19_3x3_bias_read_readvariableop/savev2_expand_20_5x5_kernel_read_readvariableop-savev2_expand_20_5x5_bias_read_readvariableop1savev2_contract_20_3x3_kernel_read_readvariableop/savev2_contract_20_3x3_bias_read_readvariableop3savev2_policy_aggregator_kernel_read_readvariableop1savev2_policy_aggregator_bias_read_readvariableop,savev2_border_off_kernel_read_readvariableop*savev2_border_off_bias_read_readvariableop,savev2_value_head_kernel_read_readvariableop*savev2_value_head_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *k
dtypesa
_2]љ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:І
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*н
_input_shapes┬
┐: :│:│:ђ:ђ:│::ђ::	 : : ::	 : : ::	 : : ::	 : : ::	 : : ::	 : : ::	 : : ::	 : : ::	 : : ::	 : : ::	 : : ::	 : : ::	 : : ::	 : : ::	 : : ::	 : : ::	 : : ::	 : : ::	 : : ::::::	т,:: : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
:│:!

_output_shapes	
:│:-)
'
_output_shapes
:ђ:!

_output_shapes	
:ђ:-)
'
_output_shapes
:│: 

_output_shapes
::-)
'
_output_shapes
:ђ: 

_output_shapes
::,	(
&
_output_shapes
:	 : 


_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::,(
&
_output_shapes
:	 : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::,(
&
_output_shapes
:	 : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::,(
&
_output_shapes
:	 : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::,(
&
_output_shapes
:	 : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::,(
&
_output_shapes
:	 : 

_output_shapes
: :,(
&
_output_shapes
: :  

_output_shapes
::,!(
&
_output_shapes
:	 : "

_output_shapes
: :,#(
&
_output_shapes
: : $

_output_shapes
::,%(
&
_output_shapes
:	 : &

_output_shapes
: :,'(
&
_output_shapes
: : (

_output_shapes
::,)(
&
_output_shapes
:	 : *

_output_shapes
: :,+(
&
_output_shapes
: : ,

_output_shapes
::,-(
&
_output_shapes
:	 : .

_output_shapes
: :,/(
&
_output_shapes
: : 0

_output_shapes
::,1(
&
_output_shapes
:	 : 2

_output_shapes
: :,3(
&
_output_shapes
: : 4

_output_shapes
::,5(
&
_output_shapes
:	 : 6

_output_shapes
: :,7(
&
_output_shapes
: : 8

_output_shapes
::,9(
&
_output_shapes
:	 : :

_output_shapes
: :,;(
&
_output_shapes
: : <

_output_shapes
::,=(
&
_output_shapes
:	 : >

_output_shapes
: :,?(
&
_output_shapes
: : @

_output_shapes
::,A(
&
_output_shapes
:	 : B

_output_shapes
: :,C(
&
_output_shapes
: : D

_output_shapes
::,E(
&
_output_shapes
:	 : F

_output_shapes
: :,G(
&
_output_shapes
: : H

_output_shapes
::,I(
&
_output_shapes
:	 : J

_output_shapes
: :,K(
&
_output_shapes
: : L

_output_shapes
::,M(
&
_output_shapes
:	 : N

_output_shapes
: :,O(
&
_output_shapes
: : P

_output_shapes
::,Q(
&
_output_shapes
:	 : R

_output_shapes
: :,S(
&
_output_shapes
: : T

_output_shapes
::,U(
&
_output_shapes
:: V

_output_shapes
::,W(
&
_output_shapes
:: X

_output_shapes
::%Y!

_output_shapes
:	т,: Z

_output_shapes
::[

_output_shapes
: :\

_output_shapes
: :]

_output_shapes
: 
Ы
q
E__inference_skip_17_layer_call_and_return_conditional_losses_10433679
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
л
j
N__inference_flat_value_input_layer_call_and_return_conditional_losses_10430007

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    e  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         т,Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         т,"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ч
Ц
0__inference_expand_12_5x5_layer_call_fn_10433311

inputs!
unknown:	 
	unknown_0: 
identityѕбStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_12_5x5_layer_call_and_return_conditional_losses_10429536w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
 
Д
2__inference_contract_15_3x3_layer_call_fn_10433526

inputs!
unknown: 
	unknown_0:
identityѕбStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_15_3x3_layer_call_and_return_conditional_losses_10429706w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
 
Д
2__inference_contract_17_3x3_layer_call_fn_10433656

inputs!
unknown: 
	unknown_0:
identityѕбStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_17_3x3_layer_call_and_return_conditional_losses_10429808w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
ь
v
L__inference_concatenate_24_layer_call_and_return_conditional_losses_10429268

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :}
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
т)
А
&__inference_signature_wrapper_10432559

inputs"
unknown:ђ
	unknown_0:	ђ$
	unknown_1:│
	unknown_2:	│$
	unknown_3:│
	unknown_4:$
	unknown_5:ђ
	unknown_6:#
	unknown_7:	 
	unknown_8: #
	unknown_9: 

unknown_10:$

unknown_11:	 

unknown_12: $

unknown_13: 

unknown_14:$

unknown_15:	 

unknown_16: $

unknown_17: 

unknown_18:$

unknown_19:	 

unknown_20: $

unknown_21: 

unknown_22:$

unknown_23:	 

unknown_24: $

unknown_25: 

unknown_26:$

unknown_27:	 

unknown_28: $

unknown_29: 

unknown_30:$

unknown_31:	 

unknown_32: $

unknown_33: 

unknown_34:$

unknown_35:	 

unknown_36: $

unknown_37: 

unknown_38:$

unknown_39:	 

unknown_40: $

unknown_41: 

unknown_42:$

unknown_43:	 

unknown_44: $

unknown_45: 

unknown_46:$

unknown_47:	 

unknown_48: $

unknown_49: 

unknown_50:$

unknown_51:	 

unknown_52: $

unknown_53: 

unknown_54:$

unknown_55:	 

unknown_56: $

unknown_57: 

unknown_58:$

unknown_59:	 

unknown_60: $

unknown_61: 

unknown_62:$

unknown_63:	 

unknown_64: $

unknown_65: 

unknown_66:$

unknown_67:	 

unknown_68: $

unknown_69: 

unknown_70:$

unknown_71:	 

unknown_72: $

unknown_73: 

unknown_74:$

unknown_75:	 

unknown_76: $

unknown_77: 

unknown_78:$

unknown_79:	 

unknown_80: $

unknown_81: 

unknown_82:$

unknown_83:

unknown_84:$

unknown_85:

unknown_86:

unknown_87:	т,

unknown_88:
identity

identity_1ѕбStatefulPartitionedCallЌ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74
unknown_75
unknown_76
unknown_77
unknown_78
unknown_79
unknown_80
unknown_81
unknown_82
unknown_83
unknown_84
unknown_85
unknown_86
unknown_87
unknown_88*f
Tin_
]2[*
Tout
2*
_collective_manager_ids
 *;
_output_shapes)
':         ж:         *|
_read_only_resource_inputs^
\Z	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ*0
config_proto 

CPU

GPU2*0J 8ѓ *,
f'R%
#__inference__wrapped_model_10428931p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         жq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*С
_input_shapesм
¤:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ж
n
D__inference_skip_3_layer_call_and_return_conditional_losses_10429106

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
л
j
N__inference_flat_value_input_layer_call_and_return_conditional_losses_10433937

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    e  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         т,Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         т,"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ш
x
L__inference_concatenate_32_layer_call_and_return_conditional_losses_10433497
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
ќ
ё
K__inference_expand_11_5x5_layer_call_and_return_conditional_losses_10433257

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:          m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
ќ
ё
K__inference_expand_11_5x5_layer_call_and_return_conditional_losses_10429485

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:          m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
ч
Ц
0__inference_expand_15_5x5_layer_call_fn_10433506

inputs!
unknown:	 
	unknown_0: 
identityѕбStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_15_5x5_layer_call_and_return_conditional_losses_10429689w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
ш
x
L__inference_concatenate_33_layer_call_and_return_conditional_losses_10433562
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
Ы
q
E__inference_skip_10_layer_call_and_return_conditional_losses_10433224
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
ж
n
D__inference_skip_6_layer_call_and_return_conditional_losses_10429259

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
Ж
o
E__inference_skip_13_layer_call_and_return_conditional_losses_10429616

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
ь
v
L__inference_concatenate_23_layer_call_and_return_conditional_losses_10429217

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :}
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
Ћ
Ѓ
J__inference_expand_4_5x5_layer_call_and_return_conditional_losses_10432802

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:          m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
Ќ
Ё
L__inference_contract_7_3x3_layer_call_and_return_conditional_losses_10433017

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
ќ
ё
K__inference_expand_15_5x5_layer_call_and_return_conditional_losses_10433517

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:          m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
ж
n
D__inference_skip_9_layer_call_and_return_conditional_losses_10429412

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
ў
є
M__inference_contract_12_3x3_layer_call_and_return_conditional_losses_10433342

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Ы
q
E__inference_skip_16_layer_call_and_return_conditional_losses_10433614
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
т
]
1__inference_concatenate_23_layer_call_fn_10432905
inputs_0
inputs_1
identity¤
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_23_layer_call_and_return_conditional_losses_10429217h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
ш
x
L__inference_concatenate_27_layer_call_and_return_conditional_losses_10433172
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
ь
v
L__inference_concatenate_21_layer_call_and_return_conditional_losses_10429115

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :}
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
ы
p
D__inference_skip_2_layer_call_and_return_conditional_losses_10432704
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
ь
v
L__inference_concatenate_34_layer_call_and_return_conditional_losses_10429778

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :}
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
ў
є
M__inference_contract_20_3x3_layer_call_and_return_conditional_losses_10433862

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
щ
ц
/__inference_expand_2_5x5_layer_call_fn_10432661

inputs!
unknown:	 
	unknown_0: 
identityѕбStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_2_5x5_layer_call_and_return_conditional_losses_10429026w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
т
]
1__inference_concatenate_35_layer_call_fn_10433685
inputs_0
inputs_1
identity¤
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_35_layer_call_and_return_conditional_losses_10429829h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
О
V
*__inference_skip_19_layer_call_fn_10433803
inputs_0
inputs_1
identity╚
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_19_layer_call_and_return_conditional_losses_10429922h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
Ћ
Ѓ
J__inference_expand_7_5x5_layer_call_and_return_conditional_losses_10432997

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:          m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
Їё
█.
M__inference_gomoku_resnet_1_layer_call_and_return_conditional_losses_10432370

inputs2
expand_1_11x11_10432099:ђ&
expand_1_11x11_10432101:	ђ6
heuristic_detector_10432104:│*
heuristic_detector_10432106:	│6
heuristic_priority_10432109:│)
heuristic_priority_10432111:2
contract_1_5x5_10432114:ђ%
contract_1_5x5_10432116:/
expand_2_5x5_10432120:	 #
expand_2_5x5_10432122: 1
contract_2_3x3_10432125: %
contract_2_3x3_10432127:/
expand_3_5x5_10432132:	 #
expand_3_5x5_10432134: 1
contract_3_3x3_10432137: %
contract_3_3x3_10432139:/
expand_4_5x5_10432144:	 #
expand_4_5x5_10432146: 1
contract_4_3x3_10432149: %
contract_4_3x3_10432151:/
expand_5_5x5_10432156:	 #
expand_5_5x5_10432158: 1
contract_5_3x3_10432161: %
contract_5_3x3_10432163:/
expand_6_5x5_10432168:	 #
expand_6_5x5_10432170: 1
contract_6_3x3_10432173: %
contract_6_3x3_10432175:/
expand_7_5x5_10432180:	 #
expand_7_5x5_10432182: 1
contract_7_3x3_10432185: %
contract_7_3x3_10432187:/
expand_8_5x5_10432192:	 #
expand_8_5x5_10432194: 1
contract_8_3x3_10432197: %
contract_8_3x3_10432199:/
expand_9_5x5_10432204:	 #
expand_9_5x5_10432206: 1
contract_9_3x3_10432209: %
contract_9_3x3_10432211:0
expand_10_5x5_10432216:	 $
expand_10_5x5_10432218: 2
contract_10_3x3_10432221: &
contract_10_3x3_10432223:0
expand_11_5x5_10432228:	 $
expand_11_5x5_10432230: 2
contract_11_3x3_10432233: &
contract_11_3x3_10432235:0
expand_12_5x5_10432240:	 $
expand_12_5x5_10432242: 2
contract_12_3x3_10432245: &
contract_12_3x3_10432247:0
expand_13_5x5_10432252:	 $
expand_13_5x5_10432254: 2
contract_13_3x3_10432257: &
contract_13_3x3_10432259:0
expand_14_5x5_10432264:	 $
expand_14_5x5_10432266: 2
contract_14_3x3_10432269: &
contract_14_3x3_10432271:0
expand_15_5x5_10432276:	 $
expand_15_5x5_10432278: 2
contract_15_3x3_10432281: &
contract_15_3x3_10432283:0
expand_16_5x5_10432288:	 $
expand_16_5x5_10432290: 2
contract_16_3x3_10432293: &
contract_16_3x3_10432295:0
expand_17_5x5_10432300:	 $
expand_17_5x5_10432302: 2
contract_17_3x3_10432305: &
contract_17_3x3_10432307:0
expand_18_5x5_10432312:	 $
expand_18_5x5_10432314: 2
contract_18_3x3_10432317: &
contract_18_3x3_10432319:0
expand_19_5x5_10432324:	 $
expand_19_5x5_10432326: 2
contract_19_3x3_10432329: &
contract_19_3x3_10432331:0
expand_20_5x5_10432336:	 $
expand_20_5x5_10432338: 2
contract_20_3x3_10432341: &
contract_20_3x3_10432343:4
policy_aggregator_10432348:(
policy_aggregator_10432350:-
border_off_10432354:!
border_off_10432356:&
value_head_10432362:	т,!
value_head_10432364:
identity

identity_1ѕб"border_off/StatefulPartitionedCallб'contract_10_3x3/StatefulPartitionedCallб'contract_11_3x3/StatefulPartitionedCallб'contract_12_3x3/StatefulPartitionedCallб'contract_13_3x3/StatefulPartitionedCallб'contract_14_3x3/StatefulPartitionedCallб'contract_15_3x3/StatefulPartitionedCallб'contract_16_3x3/StatefulPartitionedCallб'contract_17_3x3/StatefulPartitionedCallб'contract_18_3x3/StatefulPartitionedCallб'contract_19_3x3/StatefulPartitionedCallб&contract_1_5x5/StatefulPartitionedCallб'contract_20_3x3/StatefulPartitionedCallб&contract_2_3x3/StatefulPartitionedCallб&contract_3_3x3/StatefulPartitionedCallб&contract_4_3x3/StatefulPartitionedCallб&contract_5_3x3/StatefulPartitionedCallб&contract_6_3x3/StatefulPartitionedCallб&contract_7_3x3/StatefulPartitionedCallб&contract_8_3x3/StatefulPartitionedCallб&contract_9_3x3/StatefulPartitionedCallб%expand_10_5x5/StatefulPartitionedCallб%expand_11_5x5/StatefulPartitionedCallб%expand_12_5x5/StatefulPartitionedCallб%expand_13_5x5/StatefulPartitionedCallб%expand_14_5x5/StatefulPartitionedCallб%expand_15_5x5/StatefulPartitionedCallб%expand_16_5x5/StatefulPartitionedCallб%expand_17_5x5/StatefulPartitionedCallб%expand_18_5x5/StatefulPartitionedCallб%expand_19_5x5/StatefulPartitionedCallб&expand_1_11x11/StatefulPartitionedCallб%expand_20_5x5/StatefulPartitionedCallб$expand_2_5x5/StatefulPartitionedCallб$expand_3_5x5/StatefulPartitionedCallб$expand_4_5x5/StatefulPartitionedCallб$expand_5_5x5/StatefulPartitionedCallб$expand_6_5x5/StatefulPartitionedCallб$expand_7_5x5/StatefulPartitionedCallб$expand_8_5x5/StatefulPartitionedCallб$expand_9_5x5/StatefulPartitionedCallб*heuristic_detector/StatefulPartitionedCallб*heuristic_priority/StatefulPartitionedCallб)policy_aggregator/StatefulPartitionedCallб"value_head/StatefulPartitionedCallџ
&expand_1_11x11/StatefulPartitionedCallStatefulPartitionedCallinputsexpand_1_11x11_10432099expand_1_11x11_10432101*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_expand_1_11x11_layer_call_and_return_conditional_losses_10428949ф
*heuristic_detector/StatefulPartitionedCallStatefulPartitionedCallinputsheuristic_detector_10432104heuristic_detector_10432106*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         │*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_heuristic_detector_layer_call_and_return_conditional_losses_10428966о
*heuristic_priority/StatefulPartitionedCallStatefulPartitionedCall3heuristic_detector/StatefulPartitionedCall:output:0heuristic_priority_10432109heuristic_priority_10432111*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_heuristic_priority_layer_call_and_return_conditional_losses_10428983┬
&contract_1_5x5/StatefulPartitionedCallStatefulPartitionedCall/expand_1_11x11/StatefulPartitionedCall:output:0contract_1_5x5_10432114contract_1_5x5_10432116*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_1_5x5_layer_call_and_return_conditional_losses_10429000░
concatenate_19/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0/contract_1_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_19_layer_call_and_return_conditional_losses_10429013▓
$expand_2_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_19/PartitionedCall:output:0expand_2_5x5_10432120expand_2_5x5_10432122*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_2_5x5_layer_call_and_return_conditional_losses_10429026└
&contract_2_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_2_5x5/StatefulPartitionedCall:output:0contract_2_3x3_10432125contract_2_3x3_10432127*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_2_3x3_layer_call_and_return_conditional_losses_10429043ю
skip_2/PartitionedCallPartitionedCall/contract_2_3x3/StatefulPartitionedCall:output:0/contract_1_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_2_layer_call_and_return_conditional_losses_10429055а
concatenate_20/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_20_layer_call_and_return_conditional_losses_10429064▓
$expand_3_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_20/PartitionedCall:output:0expand_3_5x5_10432132expand_3_5x5_10432134*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_3_5x5_layer_call_and_return_conditional_losses_10429077└
&contract_3_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_3_5x5/StatefulPartitionedCall:output:0contract_3_3x3_10432137contract_3_3x3_10432139*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_3_3x3_layer_call_and_return_conditional_losses_10429094ї
skip_3/PartitionedCallPartitionedCall/contract_3_3x3/StatefulPartitionedCall:output:0skip_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_3_layer_call_and_return_conditional_losses_10429106а
concatenate_21/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_21_layer_call_and_return_conditional_losses_10429115▓
$expand_4_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_21/PartitionedCall:output:0expand_4_5x5_10432144expand_4_5x5_10432146*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_4_5x5_layer_call_and_return_conditional_losses_10429128└
&contract_4_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_4_5x5/StatefulPartitionedCall:output:0contract_4_3x3_10432149contract_4_3x3_10432151*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_4_3x3_layer_call_and_return_conditional_losses_10429145ї
skip_4/PartitionedCallPartitionedCall/contract_4_3x3/StatefulPartitionedCall:output:0skip_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_4_layer_call_and_return_conditional_losses_10429157а
concatenate_22/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_22_layer_call_and_return_conditional_losses_10429166▓
$expand_5_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_22/PartitionedCall:output:0expand_5_5x5_10432156expand_5_5x5_10432158*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_5_5x5_layer_call_and_return_conditional_losses_10429179└
&contract_5_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_5_5x5/StatefulPartitionedCall:output:0contract_5_3x3_10432161contract_5_3x3_10432163*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_5_3x3_layer_call_and_return_conditional_losses_10429196ї
skip_5/PartitionedCallPartitionedCall/contract_5_3x3/StatefulPartitionedCall:output:0skip_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_5_layer_call_and_return_conditional_losses_10429208а
concatenate_23/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_23_layer_call_and_return_conditional_losses_10429217▓
$expand_6_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_23/PartitionedCall:output:0expand_6_5x5_10432168expand_6_5x5_10432170*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_6_5x5_layer_call_and_return_conditional_losses_10429230└
&contract_6_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_6_5x5/StatefulPartitionedCall:output:0contract_6_3x3_10432173contract_6_3x3_10432175*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_6_3x3_layer_call_and_return_conditional_losses_10429247ї
skip_6/PartitionedCallPartitionedCall/contract_6_3x3/StatefulPartitionedCall:output:0skip_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_6_layer_call_and_return_conditional_losses_10429259а
concatenate_24/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_24_layer_call_and_return_conditional_losses_10429268▓
$expand_7_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_24/PartitionedCall:output:0expand_7_5x5_10432180expand_7_5x5_10432182*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_7_5x5_layer_call_and_return_conditional_losses_10429281└
&contract_7_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_7_5x5/StatefulPartitionedCall:output:0contract_7_3x3_10432185contract_7_3x3_10432187*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_7_3x3_layer_call_and_return_conditional_losses_10429298ї
skip_7/PartitionedCallPartitionedCall/contract_7_3x3/StatefulPartitionedCall:output:0skip_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_7_layer_call_and_return_conditional_losses_10429310а
concatenate_25/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_25_layer_call_and_return_conditional_losses_10429319▓
$expand_8_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_25/PartitionedCall:output:0expand_8_5x5_10432192expand_8_5x5_10432194*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_8_5x5_layer_call_and_return_conditional_losses_10429332└
&contract_8_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_8_5x5/StatefulPartitionedCall:output:0contract_8_3x3_10432197contract_8_3x3_10432199*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_8_3x3_layer_call_and_return_conditional_losses_10429349ї
skip_8/PartitionedCallPartitionedCall/contract_8_3x3/StatefulPartitionedCall:output:0skip_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_8_layer_call_and_return_conditional_losses_10429361а
concatenate_26/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_26_layer_call_and_return_conditional_losses_10429370▓
$expand_9_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_26/PartitionedCall:output:0expand_9_5x5_10432204expand_9_5x5_10432206*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_9_5x5_layer_call_and_return_conditional_losses_10429383└
&contract_9_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_9_5x5/StatefulPartitionedCall:output:0contract_9_3x3_10432209contract_9_3x3_10432211*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_9_3x3_layer_call_and_return_conditional_losses_10429400ї
skip_9/PartitionedCallPartitionedCall/contract_9_3x3/StatefulPartitionedCall:output:0skip_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_skip_9_layer_call_and_return_conditional_losses_10429412а
concatenate_27/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_27_layer_call_and_return_conditional_losses_10429421Х
%expand_10_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_27/PartitionedCall:output:0expand_10_5x5_10432216expand_10_5x5_10432218*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_10_5x5_layer_call_and_return_conditional_losses_10429434┼
'contract_10_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_10_5x5/StatefulPartitionedCall:output:0contract_10_3x3_10432221contract_10_3x3_10432223*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_10_3x3_layer_call_and_return_conditional_losses_10429451Ј
skip_10/PartitionedCallPartitionedCall0contract_10_3x3/StatefulPartitionedCall:output:0skip_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_10_layer_call_and_return_conditional_losses_10429463А
concatenate_28/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_28_layer_call_and_return_conditional_losses_10429472Х
%expand_11_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_28/PartitionedCall:output:0expand_11_5x5_10432228expand_11_5x5_10432230*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_11_5x5_layer_call_and_return_conditional_losses_10429485┼
'contract_11_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_11_5x5/StatefulPartitionedCall:output:0contract_11_3x3_10432233contract_11_3x3_10432235*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_11_3x3_layer_call_and_return_conditional_losses_10429502љ
skip_11/PartitionedCallPartitionedCall0contract_11_3x3/StatefulPartitionedCall:output:0 skip_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_11_layer_call_and_return_conditional_losses_10429514А
concatenate_29/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_29_layer_call_and_return_conditional_losses_10429523Х
%expand_12_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_29/PartitionedCall:output:0expand_12_5x5_10432240expand_12_5x5_10432242*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_12_5x5_layer_call_and_return_conditional_losses_10429536┼
'contract_12_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_12_5x5/StatefulPartitionedCall:output:0contract_12_3x3_10432245contract_12_3x3_10432247*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_12_3x3_layer_call_and_return_conditional_losses_10429553љ
skip_12/PartitionedCallPartitionedCall0contract_12_3x3/StatefulPartitionedCall:output:0 skip_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_12_layer_call_and_return_conditional_losses_10429565А
concatenate_30/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_30_layer_call_and_return_conditional_losses_10429574Х
%expand_13_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_30/PartitionedCall:output:0expand_13_5x5_10432252expand_13_5x5_10432254*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_13_5x5_layer_call_and_return_conditional_losses_10429587┼
'contract_13_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_13_5x5/StatefulPartitionedCall:output:0contract_13_3x3_10432257contract_13_3x3_10432259*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_13_3x3_layer_call_and_return_conditional_losses_10429604љ
skip_13/PartitionedCallPartitionedCall0contract_13_3x3/StatefulPartitionedCall:output:0 skip_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_13_layer_call_and_return_conditional_losses_10429616А
concatenate_31/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_31_layer_call_and_return_conditional_losses_10429625Х
%expand_14_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_31/PartitionedCall:output:0expand_14_5x5_10432264expand_14_5x5_10432266*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_14_5x5_layer_call_and_return_conditional_losses_10429638┼
'contract_14_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_14_5x5/StatefulPartitionedCall:output:0contract_14_3x3_10432269contract_14_3x3_10432271*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_14_3x3_layer_call_and_return_conditional_losses_10429655љ
skip_14/PartitionedCallPartitionedCall0contract_14_3x3/StatefulPartitionedCall:output:0 skip_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_14_layer_call_and_return_conditional_losses_10429667А
concatenate_32/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_14/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_32_layer_call_and_return_conditional_losses_10429676Х
%expand_15_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_32/PartitionedCall:output:0expand_15_5x5_10432276expand_15_5x5_10432278*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_15_5x5_layer_call_and_return_conditional_losses_10429689┼
'contract_15_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_15_5x5/StatefulPartitionedCall:output:0contract_15_3x3_10432281contract_15_3x3_10432283*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_15_3x3_layer_call_and_return_conditional_losses_10429706љ
skip_15/PartitionedCallPartitionedCall0contract_15_3x3/StatefulPartitionedCall:output:0 skip_14/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_15_layer_call_and_return_conditional_losses_10429718А
concatenate_33/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_33_layer_call_and_return_conditional_losses_10429727Х
%expand_16_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_33/PartitionedCall:output:0expand_16_5x5_10432288expand_16_5x5_10432290*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_16_5x5_layer_call_and_return_conditional_losses_10429740┼
'contract_16_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_16_5x5/StatefulPartitionedCall:output:0contract_16_3x3_10432293contract_16_3x3_10432295*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_16_3x3_layer_call_and_return_conditional_losses_10429757љ
skip_16/PartitionedCallPartitionedCall0contract_16_3x3/StatefulPartitionedCall:output:0 skip_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_16_layer_call_and_return_conditional_losses_10429769А
concatenate_34/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_34_layer_call_and_return_conditional_losses_10429778Х
%expand_17_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_34/PartitionedCall:output:0expand_17_5x5_10432300expand_17_5x5_10432302*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_17_5x5_layer_call_and_return_conditional_losses_10429791┼
'contract_17_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_17_5x5/StatefulPartitionedCall:output:0contract_17_3x3_10432305contract_17_3x3_10432307*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_17_3x3_layer_call_and_return_conditional_losses_10429808љ
skip_17/PartitionedCallPartitionedCall0contract_17_3x3/StatefulPartitionedCall:output:0 skip_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_17_layer_call_and_return_conditional_losses_10429820А
concatenate_35/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_35_layer_call_and_return_conditional_losses_10429829Х
%expand_18_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_35/PartitionedCall:output:0expand_18_5x5_10432312expand_18_5x5_10432314*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_18_5x5_layer_call_and_return_conditional_losses_10429842┼
'contract_18_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_18_5x5/StatefulPartitionedCall:output:0contract_18_3x3_10432317contract_18_3x3_10432319*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_18_3x3_layer_call_and_return_conditional_losses_10429859љ
skip_18/PartitionedCallPartitionedCall0contract_18_3x3/StatefulPartitionedCall:output:0 skip_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_18_layer_call_and_return_conditional_losses_10429871А
concatenate_36/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_18/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_36_layer_call_and_return_conditional_losses_10429880Х
%expand_19_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_36/PartitionedCall:output:0expand_19_5x5_10432324expand_19_5x5_10432326*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_19_5x5_layer_call_and_return_conditional_losses_10429893┼
'contract_19_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_19_5x5/StatefulPartitionedCall:output:0contract_19_3x3_10432329contract_19_3x3_10432331*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_19_3x3_layer_call_and_return_conditional_losses_10429910љ
skip_19/PartitionedCallPartitionedCall0contract_19_3x3/StatefulPartitionedCall:output:0 skip_18/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_19_layer_call_and_return_conditional_losses_10429922А
concatenate_37/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_19/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_37_layer_call_and_return_conditional_losses_10429931Х
%expand_20_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_37/PartitionedCall:output:0expand_20_5x5_10432336expand_20_5x5_10432338*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_20_5x5_layer_call_and_return_conditional_losses_10429944┼
'contract_20_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_20_5x5/StatefulPartitionedCall:output:0contract_20_3x3_10432341contract_20_3x3_10432343*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_20_3x3_layer_call_and_return_conditional_losses_10429961љ
skip_20/PartitionedCallPartitionedCall0contract_20_3x3/StatefulPartitionedCall:output:0 skip_19/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_20_layer_call_and_return_conditional_losses_10429973Д
all_value_input/PartitionedCallPartitionedCall0contract_20_3x3/StatefulPartitionedCall:output:0'concatenate_19/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_all_value_input_layer_call_and_return_conditional_losses_10429982┐
)policy_aggregator/StatefulPartitionedCallStatefulPartitionedCall skip_20/PartitionedCall:output:0policy_aggregator_10432348policy_aggregator_10432350*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_policy_aggregator_layer_call_and_return_conditional_losses_10429995­
 flat_value_input/PartitionedCallPartitionedCall(all_value_input/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         т,* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_flat_value_input_layer_call_and_return_conditional_losses_10430007х
"border_off/StatefulPartitionedCallStatefulPartitionedCall2policy_aggregator/StatefulPartitionedCall:output:0border_off_10432354border_off_10432356*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_border_off_layer_call_and_return_conditional_losses_10430019`
tf.math.truediv_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ╚Bе
tf.math.truediv_1/truedivRealDiv)flat_value_input/PartitionedCall:output:0$tf.math.truediv_1/truediv/y:output:0*
T0*(
_output_shapes
:         т,ж
flat_logits/PartitionedCallPartitionedCall+border_off/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ж* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_flat_logits_layer_call_and_return_conditional_losses_10430033ў
"value_head/StatefulPartitionedCallStatefulPartitionedCalltf.math.truediv_1/truediv:z:0value_head_10432362value_head_10432364*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_value_head_layer_call_and_return_conditional_losses_10430046Р
policy_head/PartitionedCallPartitionedCall$flat_logits/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ж* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_policy_head_layer_call_and_return_conditional_losses_10430057t
IdentityIdentity$policy_head/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ж|

Identity_1Identity+value_head/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Ь
NoOpNoOp#^border_off/StatefulPartitionedCall(^contract_10_3x3/StatefulPartitionedCall(^contract_11_3x3/StatefulPartitionedCall(^contract_12_3x3/StatefulPartitionedCall(^contract_13_3x3/StatefulPartitionedCall(^contract_14_3x3/StatefulPartitionedCall(^contract_15_3x3/StatefulPartitionedCall(^contract_16_3x3/StatefulPartitionedCall(^contract_17_3x3/StatefulPartitionedCall(^contract_18_3x3/StatefulPartitionedCall(^contract_19_3x3/StatefulPartitionedCall'^contract_1_5x5/StatefulPartitionedCall(^contract_20_3x3/StatefulPartitionedCall'^contract_2_3x3/StatefulPartitionedCall'^contract_3_3x3/StatefulPartitionedCall'^contract_4_3x3/StatefulPartitionedCall'^contract_5_3x3/StatefulPartitionedCall'^contract_6_3x3/StatefulPartitionedCall'^contract_7_3x3/StatefulPartitionedCall'^contract_8_3x3/StatefulPartitionedCall'^contract_9_3x3/StatefulPartitionedCall&^expand_10_5x5/StatefulPartitionedCall&^expand_11_5x5/StatefulPartitionedCall&^expand_12_5x5/StatefulPartitionedCall&^expand_13_5x5/StatefulPartitionedCall&^expand_14_5x5/StatefulPartitionedCall&^expand_15_5x5/StatefulPartitionedCall&^expand_16_5x5/StatefulPartitionedCall&^expand_17_5x5/StatefulPartitionedCall&^expand_18_5x5/StatefulPartitionedCall&^expand_19_5x5/StatefulPartitionedCall'^expand_1_11x11/StatefulPartitionedCall&^expand_20_5x5/StatefulPartitionedCall%^expand_2_5x5/StatefulPartitionedCall%^expand_3_5x5/StatefulPartitionedCall%^expand_4_5x5/StatefulPartitionedCall%^expand_5_5x5/StatefulPartitionedCall%^expand_6_5x5/StatefulPartitionedCall%^expand_7_5x5/StatefulPartitionedCall%^expand_8_5x5/StatefulPartitionedCall%^expand_9_5x5/StatefulPartitionedCall+^heuristic_detector/StatefulPartitionedCall+^heuristic_priority/StatefulPartitionedCall*^policy_aggregator/StatefulPartitionedCall#^value_head/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*С
_input_shapesм
¤:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"border_off/StatefulPartitionedCall"border_off/StatefulPartitionedCall2R
'contract_10_3x3/StatefulPartitionedCall'contract_10_3x3/StatefulPartitionedCall2R
'contract_11_3x3/StatefulPartitionedCall'contract_11_3x3/StatefulPartitionedCall2R
'contract_12_3x3/StatefulPartitionedCall'contract_12_3x3/StatefulPartitionedCall2R
'contract_13_3x3/StatefulPartitionedCall'contract_13_3x3/StatefulPartitionedCall2R
'contract_14_3x3/StatefulPartitionedCall'contract_14_3x3/StatefulPartitionedCall2R
'contract_15_3x3/StatefulPartitionedCall'contract_15_3x3/StatefulPartitionedCall2R
'contract_16_3x3/StatefulPartitionedCall'contract_16_3x3/StatefulPartitionedCall2R
'contract_17_3x3/StatefulPartitionedCall'contract_17_3x3/StatefulPartitionedCall2R
'contract_18_3x3/StatefulPartitionedCall'contract_18_3x3/StatefulPartitionedCall2R
'contract_19_3x3/StatefulPartitionedCall'contract_19_3x3/StatefulPartitionedCall2P
&contract_1_5x5/StatefulPartitionedCall&contract_1_5x5/StatefulPartitionedCall2R
'contract_20_3x3/StatefulPartitionedCall'contract_20_3x3/StatefulPartitionedCall2P
&contract_2_3x3/StatefulPartitionedCall&contract_2_3x3/StatefulPartitionedCall2P
&contract_3_3x3/StatefulPartitionedCall&contract_3_3x3/StatefulPartitionedCall2P
&contract_4_3x3/StatefulPartitionedCall&contract_4_3x3/StatefulPartitionedCall2P
&contract_5_3x3/StatefulPartitionedCall&contract_5_3x3/StatefulPartitionedCall2P
&contract_6_3x3/StatefulPartitionedCall&contract_6_3x3/StatefulPartitionedCall2P
&contract_7_3x3/StatefulPartitionedCall&contract_7_3x3/StatefulPartitionedCall2P
&contract_8_3x3/StatefulPartitionedCall&contract_8_3x3/StatefulPartitionedCall2P
&contract_9_3x3/StatefulPartitionedCall&contract_9_3x3/StatefulPartitionedCall2N
%expand_10_5x5/StatefulPartitionedCall%expand_10_5x5/StatefulPartitionedCall2N
%expand_11_5x5/StatefulPartitionedCall%expand_11_5x5/StatefulPartitionedCall2N
%expand_12_5x5/StatefulPartitionedCall%expand_12_5x5/StatefulPartitionedCall2N
%expand_13_5x5/StatefulPartitionedCall%expand_13_5x5/StatefulPartitionedCall2N
%expand_14_5x5/StatefulPartitionedCall%expand_14_5x5/StatefulPartitionedCall2N
%expand_15_5x5/StatefulPartitionedCall%expand_15_5x5/StatefulPartitionedCall2N
%expand_16_5x5/StatefulPartitionedCall%expand_16_5x5/StatefulPartitionedCall2N
%expand_17_5x5/StatefulPartitionedCall%expand_17_5x5/StatefulPartitionedCall2N
%expand_18_5x5/StatefulPartitionedCall%expand_18_5x5/StatefulPartitionedCall2N
%expand_19_5x5/StatefulPartitionedCall%expand_19_5x5/StatefulPartitionedCall2P
&expand_1_11x11/StatefulPartitionedCall&expand_1_11x11/StatefulPartitionedCall2N
%expand_20_5x5/StatefulPartitionedCall%expand_20_5x5/StatefulPartitionedCall2L
$expand_2_5x5/StatefulPartitionedCall$expand_2_5x5/StatefulPartitionedCall2L
$expand_3_5x5/StatefulPartitionedCall$expand_3_5x5/StatefulPartitionedCall2L
$expand_4_5x5/StatefulPartitionedCall$expand_4_5x5/StatefulPartitionedCall2L
$expand_5_5x5/StatefulPartitionedCall$expand_5_5x5/StatefulPartitionedCall2L
$expand_6_5x5/StatefulPartitionedCall$expand_6_5x5/StatefulPartitionedCall2L
$expand_7_5x5/StatefulPartitionedCall$expand_7_5x5/StatefulPartitionedCall2L
$expand_8_5x5/StatefulPartitionedCall$expand_8_5x5/StatefulPartitionedCall2L
$expand_9_5x5/StatefulPartitionedCall$expand_9_5x5/StatefulPartitionedCall2X
*heuristic_detector/StatefulPartitionedCall*heuristic_detector/StatefulPartitionedCall2X
*heuristic_priority/StatefulPartitionedCall*heuristic_priority/StatefulPartitionedCall2V
)policy_aggregator/StatefulPartitionedCall)policy_aggregator/StatefulPartitionedCall2H
"value_head/StatefulPartitionedCall"value_head/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
 
Д
2__inference_contract_19_3x3_layer_call_fn_10433786

inputs!
unknown: 
	unknown_0:
identityѕбStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_19_3x3_layer_call_and_return_conditional_losses_10429910w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
ь
v
L__inference_concatenate_30_layer_call_and_return_conditional_losses_10429574

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :}
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
щ
ц
/__inference_expand_3_5x5_layer_call_fn_10432726

inputs!
unknown:	 
	unknown_0: 
identityѕбStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_3_5x5_layer_call_and_return_conditional_losses_10429077w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
Ж
o
E__inference_skip_16_layer_call_and_return_conditional_losses_10429769

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
Ћ
Ѓ
J__inference_expand_4_5x5_layer_call_and_return_conditional_losses_10429128

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:          m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
т
]
1__inference_concatenate_21_layer_call_fn_10432775
inputs_0
inputs_1
identity¤
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_21_layer_call_and_return_conditional_losses_10429115h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
ы
p
D__inference_skip_4_layer_call_and_return_conditional_losses_10432834
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
Ћ
Ѓ
J__inference_expand_7_5x5_layer_call_and_return_conditional_losses_10429281

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:          m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
щ
ц
/__inference_expand_6_5x5_layer_call_fn_10432921

inputs!
unknown:	 
	unknown_0: 
identityѕбStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_expand_6_5x5_layer_call_and_return_conditional_losses_10429230w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
Ы
q
E__inference_skip_12_layer_call_and_return_conditional_losses_10433354
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
О
V
*__inference_skip_15_layer_call_fn_10433543
inputs_0
inputs_1
identity╚
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_skip_15_layer_call_and_return_conditional_losses_10429718h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
т
]
1__inference_concatenate_34_layer_call_fn_10433620
inputs_0
inputs_1
identity¤
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_34_layer_call_and_return_conditional_losses_10429778h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
╦
e
I__inference_flat_logits_layer_call_and_return_conditional_losses_10433948

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    i  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         жY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ж"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ў
є
M__inference_contract_10_3x3_layer_call_and_return_conditional_losses_10429451

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
 
Д
2__inference_contract_16_3x3_layer_call_fn_10433591

inputs!
unknown: 
	unknown_0:
identityѕбStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_16_3x3_layer_call_and_return_conditional_losses_10429757w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
ш
x
L__inference_concatenate_36_layer_call_and_return_conditional_losses_10433757
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
ч
Ц
0__inference_expand_18_5x5_layer_call_fn_10433701

inputs!
unknown:	 
	unknown_0: 
identityѕбStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_18_5x5_layer_call_and_return_conditional_losses_10429842w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
Ќ
Ё
L__inference_contract_9_3x3_layer_call_and_return_conditional_losses_10433147

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
т
]
1__inference_concatenate_37_layer_call_fn_10433815
inputs_0
inputs_1
identity¤
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_37_layer_call_and_return_conditional_losses_10429931h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
Ж
o
E__inference_skip_12_layer_call_and_return_conditional_losses_10429565

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
§
д
1__inference_contract_4_3x3_layer_call_fn_10432811

inputs!
unknown: 
	unknown_0:
identityѕбStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_contract_4_3x3_layer_call_and_return_conditional_losses_10429145w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
т
]
1__inference_concatenate_30_layer_call_fn_10433360
inputs_0
inputs_1
identity¤
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_30_layer_call_and_return_conditional_losses_10429574h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
ь
v
L__inference_concatenate_35_layer_call_and_return_conditional_losses_10429829

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :}
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
ў
є
M__inference_contract_10_3x3_layer_call_and_return_conditional_losses_10433212

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
ь
v
L__inference_concatenate_29_layer_call_and_return_conditional_losses_10429523

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :}
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
т
]
1__inference_concatenate_32_layer_call_fn_10433490
inputs_0
inputs_1
identity¤
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_concatenate_32_layer_call_and_return_conditional_losses_10429676h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
ы
p
D__inference_skip_8_layer_call_and_return_conditional_losses_10433094
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
 
Д
2__inference_contract_14_3x3_layer_call_fn_10433461

inputs!
unknown: 
	unknown_0:
identityѕбStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_14_3x3_layer_call_and_return_conditional_losses_10429655w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
ч
Ц
0__inference_expand_16_5x5_layer_call_fn_10433571

inputs!
unknown:	 
	unknown_0: 
identityѕбStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_16_5x5_layer_call_and_return_conditional_losses_10429740w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
ч
Ц
0__inference_expand_11_5x5_layer_call_fn_10433246

inputs!
unknown:	 
	unknown_0: 
identityѕбStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_11_5x5_layer_call_and_return_conditional_losses_10429485w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
ў
є
M__inference_contract_18_3x3_layer_call_and_return_conditional_losses_10433732

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Ќ
Ё
L__inference_contract_2_3x3_layer_call_and_return_conditional_losses_10429043

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Ќ
Ё
L__inference_contract_6_3x3_layer_call_and_return_conditional_losses_10429247

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:         m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
ь
v
L__inference_concatenate_25_layer_call_and_return_conditional_losses_10429319

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :}
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:         	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
Ы
q
E__inference_skip_13_layer_call_and_return_conditional_losses_10433419
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1
Ж
o
E__inference_skip_20_layer_call_and_return_conditional_losses_10429973

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         :         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
 
Д
2__inference_contract_18_3x3_layer_call_fn_10433721

inputs!
unknown: 
	unknown_0:
identityѕбStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_contract_18_3x3_layer_call_and_return_conditional_losses_10429859w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
ч
Ц
0__inference_expand_19_5x5_layer_call_fn_10433766

inputs!
unknown:	 
	unknown_0: 
identityѕбStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_expand_19_5x5_layer_call_and_return_conditional_losses_10429893w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs"┐L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ш
serving_defaultр
A
inputs7
serving_default_inputs:0         @
policy_head1
StatefulPartitionedCall:0         ж>

value_head0
StatefulPartitionedCall:1         tensorflow/serving/predict:иј
Ш
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
layer-12
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
layer-16
layer-17
layer_with_weights-10
layer-18
layer_with_weights-11
layer-19
layer-20
layer-21
layer_with_weights-12
layer-22
layer_with_weights-13
layer-23
layer-24
layer-25
layer_with_weights-14
layer-26
layer_with_weights-15
layer-27
layer-28
layer-29
layer_with_weights-16
layer-30
 layer_with_weights-17
 layer-31
!layer-32
"layer-33
#layer_with_weights-18
#layer-34
$layer_with_weights-19
$layer-35
%layer-36
&layer-37
'layer_with_weights-20
'layer-38
(layer_with_weights-21
(layer-39
)layer-40
*layer-41
+layer_with_weights-22
+layer-42
,layer_with_weights-23
,layer-43
-layer-44
.layer-45
/layer_with_weights-24
/layer-46
0layer_with_weights-25
0layer-47
1layer-48
2layer-49
3layer_with_weights-26
3layer-50
4layer_with_weights-27
4layer-51
5layer-52
6layer-53
7layer_with_weights-28
7layer-54
8layer_with_weights-29
8layer-55
9layer-56
:layer-57
;layer_with_weights-30
;layer-58
<layer_with_weights-31
<layer-59
=layer-60
>layer-61
?layer_with_weights-32
?layer-62
@layer_with_weights-33
@layer-63
Alayer-64
Blayer-65
Clayer_with_weights-34
Clayer-66
Dlayer_with_weights-35
Dlayer-67
Elayer-68
Flayer-69
Glayer_with_weights-36
Glayer-70
Hlayer_with_weights-37
Hlayer-71
Ilayer-72
Jlayer-73
Klayer_with_weights-38
Klayer-74
Llayer_with_weights-39
Llayer-75
Mlayer-76
Nlayer-77
Olayer_with_weights-40
Olayer-78
Player_with_weights-41
Player-79
Qlayer-80
Rlayer_with_weights-42
Rlayer-81
Slayer-82
Tlayer_with_weights-43
Tlayer-83
Ulayer-84
Vlayer-85
Wlayer-86
Xlayer-87
Ylayer_with_weights-44
Ylayer-88
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses
`_default_save_signature
a	optimizer
bloss
c
signatures"
_tf_keras_network
"
_tf_keras_input_layer
П
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses

jkernel
kbias
 l_jit_compiled_convolution_op"
_tf_keras_layer
П
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses

skernel
tbias
 u_jit_compiled_convolution_op"
_tf_keras_layer
П
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses

|kernel
}bias
 ~_jit_compiled_convolution_op"
_tf_keras_layer
т
	variables
ђtrainable_variables
Ђregularization_losses
ѓ	keras_api
Ѓ__call__
+ё&call_and_return_all_conditional_losses
Ёkernel
	єbias
!Є_jit_compiled_convolution_op"
_tf_keras_layer
Ф
ѕ	variables
Ѕtrainable_variables
іregularization_losses
І	keras_api
ї__call__
+Ї&call_and_return_all_conditional_losses"
_tf_keras_layer
Т
ј	variables
Јtrainable_variables
љregularization_losses
Љ	keras_api
њ__call__
+Њ&call_and_return_all_conditional_losses
ћkernel
	Ћbias
!ќ_jit_compiled_convolution_op"
_tf_keras_layer
Т
Ќ	variables
ўtrainable_variables
Ўregularization_losses
џ	keras_api
Џ__call__
+ю&call_and_return_all_conditional_losses
Юkernel
	ъbias
!Ъ_jit_compiled_convolution_op"
_tf_keras_layer
Ф
а	variables
Аtrainable_variables
бregularization_losses
Б	keras_api
ц__call__
+Ц&call_and_return_all_conditional_losses"
_tf_keras_layer
Ф
д	variables
Дtrainable_variables
еregularization_losses
Е	keras_api
ф__call__
+Ф&call_and_return_all_conditional_losses"
_tf_keras_layer
Т
г	variables
Гtrainable_variables
«regularization_losses
»	keras_api
░__call__
+▒&call_and_return_all_conditional_losses
▓kernel
	│bias
!┤_jit_compiled_convolution_op"
_tf_keras_layer
Т
х	variables
Хtrainable_variables
иregularization_losses
И	keras_api
╣__call__
+║&call_and_return_all_conditional_losses
╗kernel
	╝bias
!й_jit_compiled_convolution_op"
_tf_keras_layer
Ф
Й	variables
┐trainable_variables
└regularization_losses
┴	keras_api
┬__call__
+├&call_and_return_all_conditional_losses"
_tf_keras_layer
Ф
─	variables
┼trainable_variables
кregularization_losses
К	keras_api
╚__call__
+╔&call_and_return_all_conditional_losses"
_tf_keras_layer
Т
╩	variables
╦trainable_variables
╠regularization_losses
═	keras_api
╬__call__
+¤&call_and_return_all_conditional_losses
лkernel
	Лbias
!м_jit_compiled_convolution_op"
_tf_keras_layer
Т
М	variables
нtrainable_variables
Нregularization_losses
о	keras_api
О__call__
+п&call_and_return_all_conditional_losses
┘kernel
	┌bias
!█_jit_compiled_convolution_op"
_tf_keras_layer
Ф
▄	variables
Пtrainable_variables
яregularization_losses
▀	keras_api
Я__call__
+р&call_and_return_all_conditional_losses"
_tf_keras_layer
Ф
Р	variables
сtrainable_variables
Сregularization_losses
т	keras_api
Т__call__
+у&call_and_return_all_conditional_losses"
_tf_keras_layer
Т
У	variables
жtrainable_variables
Жregularization_losses
в	keras_api
В__call__
+ь&call_and_return_all_conditional_losses
Ьkernel
	№bias
!­_jit_compiled_convolution_op"
_tf_keras_layer
Т
ы	variables
Ыtrainable_variables
зregularization_losses
З	keras_api
ш__call__
+Ш&call_and_return_all_conditional_losses
эkernel
	Эbias
!щ_jit_compiled_convolution_op"
_tf_keras_layer
Ф
Щ	variables
чtrainable_variables
Чregularization_losses
§	keras_api
■__call__
+ &call_and_return_all_conditional_losses"
_tf_keras_layer
Ф
ђ	variables
Ђtrainable_variables
ѓregularization_losses
Ѓ	keras_api
ё__call__
+Ё&call_and_return_all_conditional_losses"
_tf_keras_layer
Т
є	variables
Єtrainable_variables
ѕregularization_losses
Ѕ	keras_api
і__call__
+І&call_and_return_all_conditional_losses
їkernel
	Їbias
!ј_jit_compiled_convolution_op"
_tf_keras_layer
Т
Ј	variables
љtrainable_variables
Љregularization_losses
њ	keras_api
Њ__call__
+ћ&call_and_return_all_conditional_losses
Ћkernel
	ќbias
!Ќ_jit_compiled_convolution_op"
_tf_keras_layer
Ф
ў	variables
Ўtrainable_variables
џregularization_losses
Џ	keras_api
ю__call__
+Ю&call_and_return_all_conditional_losses"
_tf_keras_layer
Ф
ъ	variables
Ъtrainable_variables
аregularization_losses
А	keras_api
б__call__
+Б&call_and_return_all_conditional_losses"
_tf_keras_layer
Т
ц	variables
Цtrainable_variables
дregularization_losses
Д	keras_api
е__call__
+Е&call_and_return_all_conditional_losses
фkernel
	Фbias
!г_jit_compiled_convolution_op"
_tf_keras_layer
Т
Г	variables
«trainable_variables
»regularization_losses
░	keras_api
▒__call__
+▓&call_and_return_all_conditional_losses
│kernel
	┤bias
!х_jit_compiled_convolution_op"
_tf_keras_layer
Ф
Х	variables
иtrainable_variables
Иregularization_losses
╣	keras_api
║__call__
+╗&call_and_return_all_conditional_losses"
_tf_keras_layer
Ф
╝	variables
йtrainable_variables
Йregularization_losses
┐	keras_api
└__call__
+┴&call_and_return_all_conditional_losses"
_tf_keras_layer
Т
┬	variables
├trainable_variables
─regularization_losses
┼	keras_api
к__call__
+К&call_and_return_all_conditional_losses
╚kernel
	╔bias
!╩_jit_compiled_convolution_op"
_tf_keras_layer
Т
╦	variables
╠trainable_variables
═regularization_losses
╬	keras_api
¤__call__
+л&call_and_return_all_conditional_losses
Лkernel
	мbias
!М_jit_compiled_convolution_op"
_tf_keras_layer
Ф
н	variables
Нtrainable_variables
оregularization_losses
О	keras_api
п__call__
+┘&call_and_return_all_conditional_losses"
_tf_keras_layer
Ф
┌	variables
█trainable_variables
▄regularization_losses
П	keras_api
я__call__
+▀&call_and_return_all_conditional_losses"
_tf_keras_layer
Т
Я	variables
рtrainable_variables
Рregularization_losses
с	keras_api
С__call__
+т&call_and_return_all_conditional_losses
Тkernel
	уbias
!У_jit_compiled_convolution_op"
_tf_keras_layer
Т
ж	variables
Жtrainable_variables
вregularization_losses
В	keras_api
ь__call__
+Ь&call_and_return_all_conditional_losses
№kernel
	­bias
!ы_jit_compiled_convolution_op"
_tf_keras_layer
Ф
Ы	variables
зtrainable_variables
Зregularization_losses
ш	keras_api
Ш__call__
+э&call_and_return_all_conditional_losses"
_tf_keras_layer
Ф
Э	variables
щtrainable_variables
Щregularization_losses
ч	keras_api
Ч__call__
+§&call_and_return_all_conditional_losses"
_tf_keras_layer
Т
■	variables
 trainable_variables
ђregularization_losses
Ђ	keras_api
ѓ__call__
+Ѓ&call_and_return_all_conditional_losses
ёkernel
	Ёbias
!є_jit_compiled_convolution_op"
_tf_keras_layer
Т
Є	variables
ѕtrainable_variables
Ѕregularization_losses
і	keras_api
І__call__
+ї&call_and_return_all_conditional_losses
Їkernel
	јbias
!Ј_jit_compiled_convolution_op"
_tf_keras_layer
Ф
љ	variables
Љtrainable_variables
њregularization_losses
Њ	keras_api
ћ__call__
+Ћ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ф
ќ	variables
Ќtrainable_variables
ўregularization_losses
Ў	keras_api
џ__call__
+Џ&call_and_return_all_conditional_losses"
_tf_keras_layer
Т
ю	variables
Юtrainable_variables
ъregularization_losses
Ъ	keras_api
а__call__
+А&call_and_return_all_conditional_losses
бkernel
	Бbias
!ц_jit_compiled_convolution_op"
_tf_keras_layer
Т
Ц	variables
дtrainable_variables
Дregularization_losses
е	keras_api
Е__call__
+ф&call_and_return_all_conditional_losses
Фkernel
	гbias
!Г_jit_compiled_convolution_op"
_tf_keras_layer
Ф
«	variables
»trainable_variables
░regularization_losses
▒	keras_api
▓__call__
+│&call_and_return_all_conditional_losses"
_tf_keras_layer
Ф
┤	variables
хtrainable_variables
Хregularization_losses
и	keras_api
И__call__
+╣&call_and_return_all_conditional_losses"
_tf_keras_layer
Т
║	variables
╗trainable_variables
╝regularization_losses
й	keras_api
Й__call__
+┐&call_and_return_all_conditional_losses
└kernel
	┴bias
!┬_jit_compiled_convolution_op"
_tf_keras_layer
Т
├	variables
─trainable_variables
┼regularization_losses
к	keras_api
К__call__
+╚&call_and_return_all_conditional_losses
╔kernel
	╩bias
!╦_jit_compiled_convolution_op"
_tf_keras_layer
Ф
╠	variables
═trainable_variables
╬regularization_losses
¤	keras_api
л__call__
+Л&call_and_return_all_conditional_losses"
_tf_keras_layer
Ф
м	variables
Мtrainable_variables
нregularization_losses
Н	keras_api
о__call__
+О&call_and_return_all_conditional_losses"
_tf_keras_layer
Т
п	variables
┘trainable_variables
┌regularization_losses
█	keras_api
▄__call__
+П&call_and_return_all_conditional_losses
яkernel
	▀bias
!Я_jit_compiled_convolution_op"
_tf_keras_layer
Т
р	variables
Рtrainable_variables
сregularization_losses
С	keras_api
т__call__
+Т&call_and_return_all_conditional_losses
уkernel
	Уbias
!ж_jit_compiled_convolution_op"
_tf_keras_layer
Ф
Ж	variables
вtrainable_variables
Вregularization_losses
ь	keras_api
Ь__call__
+№&call_and_return_all_conditional_losses"
_tf_keras_layer
Ф
­	variables
ыtrainable_variables
Ыregularization_losses
з	keras_api
З__call__
+ш&call_and_return_all_conditional_losses"
_tf_keras_layer
Т
Ш	variables
эtrainable_variables
Эregularization_losses
щ	keras_api
Щ__call__
+ч&call_and_return_all_conditional_losses
Чkernel
	§bias
!■_jit_compiled_convolution_op"
_tf_keras_layer
Т
 	variables
ђtrainable_variables
Ђregularization_losses
ѓ	keras_api
Ѓ__call__
+ё&call_and_return_all_conditional_losses
Ёkernel
	єbias
!Є_jit_compiled_convolution_op"
_tf_keras_layer
Ф
ѕ	variables
Ѕtrainable_variables
іregularization_losses
І	keras_api
ї__call__
+Ї&call_and_return_all_conditional_losses"
_tf_keras_layer
Ф
ј	variables
Јtrainable_variables
љregularization_losses
Љ	keras_api
њ__call__
+Њ&call_and_return_all_conditional_losses"
_tf_keras_layer
Т
ћ	variables
Ћtrainable_variables
ќregularization_losses
Ќ	keras_api
ў__call__
+Ў&call_and_return_all_conditional_losses
џkernel
	Џbias
!ю_jit_compiled_convolution_op"
_tf_keras_layer
Т
Ю	variables
ъtrainable_variables
Ъregularization_losses
а	keras_api
А__call__
+б&call_and_return_all_conditional_losses
Бkernel
	цbias
!Ц_jit_compiled_convolution_op"
_tf_keras_layer
Ф
д	variables
Дtrainable_variables
еregularization_losses
Е	keras_api
ф__call__
+Ф&call_and_return_all_conditional_losses"
_tf_keras_layer
Ф
г	variables
Гtrainable_variables
«regularization_losses
»	keras_api
░__call__
+▒&call_and_return_all_conditional_losses"
_tf_keras_layer
Т
▓	variables
│trainable_variables
┤regularization_losses
х	keras_api
Х__call__
+и&call_and_return_all_conditional_losses
Иkernel
	╣bias
!║_jit_compiled_convolution_op"
_tf_keras_layer
Т
╗	variables
╝trainable_variables
йregularization_losses
Й	keras_api
┐__call__
+└&call_and_return_all_conditional_losses
┴kernel
	┬bias
!├_jit_compiled_convolution_op"
_tf_keras_layer
Ф
─	variables
┼trainable_variables
кregularization_losses
К	keras_api
╚__call__
+╔&call_and_return_all_conditional_losses"
_tf_keras_layer
Ф
╩	variables
╦trainable_variables
╠regularization_losses
═	keras_api
╬__call__
+¤&call_and_return_all_conditional_losses"
_tf_keras_layer
Т
л	variables
Лtrainable_variables
мregularization_losses
М	keras_api
н__call__
+Н&call_and_return_all_conditional_losses
оkernel
	Оbias
!п_jit_compiled_convolution_op"
_tf_keras_layer
Т
┘	variables
┌trainable_variables
█regularization_losses
▄	keras_api
П__call__
+я&call_and_return_all_conditional_losses
▀kernel
	Яbias
!р_jit_compiled_convolution_op"
_tf_keras_layer
Ф
Р	variables
сtrainable_variables
Сregularization_losses
т	keras_api
Т__call__
+у&call_and_return_all_conditional_losses"
_tf_keras_layer
Ф
У	variables
жtrainable_variables
Жregularization_losses
в	keras_api
В__call__
+ь&call_and_return_all_conditional_losses"
_tf_keras_layer
Т
Ь	variables
№trainable_variables
­regularization_losses
ы	keras_api
Ы__call__
+з&call_and_return_all_conditional_losses
Зkernel
	шbias
!Ш_jit_compiled_convolution_op"
_tf_keras_layer
Т
э	variables
Эtrainable_variables
щregularization_losses
Щ	keras_api
ч__call__
+Ч&call_and_return_all_conditional_losses
§kernel
	■bias
! _jit_compiled_convolution_op"
_tf_keras_layer
Ф
ђ	variables
Ђtrainable_variables
ѓregularization_losses
Ѓ	keras_api
ё__call__
+Ё&call_and_return_all_conditional_losses"
_tf_keras_layer
Ф
є	variables
Єtrainable_variables
ѕregularization_losses
Ѕ	keras_api
і__call__
+І&call_and_return_all_conditional_losses"
_tf_keras_layer
Т
ї	variables
Їtrainable_variables
јregularization_losses
Ј	keras_api
љ__call__
+Љ&call_and_return_all_conditional_losses
њkernel
	Њbias
!ћ_jit_compiled_convolution_op"
_tf_keras_layer
Т
Ћ	variables
ќtrainable_variables
Ќregularization_losses
ў	keras_api
Ў__call__
+џ&call_and_return_all_conditional_losses
Џkernel
	юbias
!Ю_jit_compiled_convolution_op"
_tf_keras_layer
Ф
ъ	variables
Ъtrainable_variables
аregularization_losses
А	keras_api
б__call__
+Б&call_and_return_all_conditional_losses"
_tf_keras_layer
Ф
ц	variables
Цtrainable_variables
дregularization_losses
Д	keras_api
е__call__
+Е&call_and_return_all_conditional_losses"
_tf_keras_layer
Т
ф	variables
Фtrainable_variables
гregularization_losses
Г	keras_api
«__call__
+»&call_and_return_all_conditional_losses
░kernel
	▒bias
!▓_jit_compiled_convolution_op"
_tf_keras_layer
Т
│	variables
┤trainable_variables
хregularization_losses
Х	keras_api
и__call__
+И&call_and_return_all_conditional_losses
╣kernel
	║bias
!╗_jit_compiled_convolution_op"
_tf_keras_layer
Ф
╝	variables
йtrainable_variables
Йregularization_losses
┐	keras_api
└__call__
+┴&call_and_return_all_conditional_losses"
_tf_keras_layer
Т
┬	variables
├trainable_variables
─regularization_losses
┼	keras_api
к__call__
+К&call_and_return_all_conditional_losses
╚kernel
	╔bias
!╩_jit_compiled_convolution_op"
_tf_keras_layer
Ф
╦	variables
╠trainable_variables
═regularization_losses
╬	keras_api
¤__call__
+л&call_and_return_all_conditional_losses"
_tf_keras_layer
Т
Л	variables
мtrainable_variables
Мregularization_losses
н	keras_api
Н__call__
+о&call_and_return_all_conditional_losses
Оkernel
	пbias
!┘_jit_compiled_convolution_op"
_tf_keras_layer
Ф
┌	variables
█trainable_variables
▄regularization_losses
П	keras_api
я__call__
+▀&call_and_return_all_conditional_losses"
_tf_keras_layer
Ф
Я	variables
рtrainable_variables
Рregularization_losses
с	keras_api
С__call__
+т&call_and_return_all_conditional_losses"
_tf_keras_layer
)
Т	keras_api"
_tf_keras_layer
Ф
у	variables
Уtrainable_variables
жregularization_losses
Ж	keras_api
в__call__
+В&call_and_return_all_conditional_losses"
_tf_keras_layer
├
ь	variables
Ьtrainable_variables
№regularization_losses
­	keras_api
ы__call__
+Ы&call_and_return_all_conditional_losses
зkernel
	Зbias"
_tf_keras_layer
║
j0
k1
s2
t3
|4
}5
Ё6
є7
ћ8
Ћ9
Ю10
ъ11
▓12
│13
╗14
╝15
л16
Л17
┘18
┌19
Ь20
№21
э22
Э23
ї24
Ї25
Ћ26
ќ27
ф28
Ф29
│30
┤31
╚32
╔33
Л34
м35
Т36
у37
№38
­39
ё40
Ё41
Ї42
ј43
б44
Б45
Ф46
г47
└48
┴49
╔50
╩51
я52
▀53
у54
У55
Ч56
§57
Ё58
є59
џ60
Џ61
Б62
ц63
И64
╣65
┴66
┬67
о68
О69
▀70
Я71
З72
ш73
§74
■75
њ76
Њ77
Џ78
ю79
░80
▒81
╣82
║83
╚84
╔85
О86
п87
з88
З89"
trackable_list_wrapper
ѕ
s0
t1
Ё2
є3
ћ4
Ћ5
Ю6
ъ7
▓8
│9
╗10
╝11
л12
Л13
┘14
┌15
Ь16
№17
э18
Э19
ї20
Ї21
Ћ22
ќ23
ф24
Ф25
│26
┤27
╚28
╔29
Л30
м31
Т32
у33
№34
­35
ё36
Ё37
Ї38
ј39
б40
Б41
Ф42
г43
└44
┴45
╔46
╩47
я48
▀49
у50
У51
Ч52
§53
Ё54
є55
џ56
Џ57
Б58
ц59
И60
╣61
┴62
┬63
о64
О65
▀66
Я67
З68
ш69
§70
■71
њ72
Њ73
Џ74
ю75
░76
▒77
╣78
║79
╚80
╔81
з82
З83"
trackable_list_wrapper
 "
trackable_list_wrapper
¤
шnon_trainable_variables
Шlayers
эmetrics
 Эlayer_regularization_losses
щlayer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
`_default_save_signature
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
Т
Щtrace_0
чtrace_12Ф
2__inference_gomoku_resnet_1_layer_call_fn_10430246
2__inference_gomoku_resnet_1_layer_call_fn_10431822└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 zЩtrace_0zчtrace_1
ю
Чtrace_0
§trace_12р
M__inference_gomoku_resnet_1_layer_call_and_return_conditional_losses_10432096
M__inference_gomoku_resnet_1_layer_call_and_return_conditional_losses_10432370└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 zЧtrace_0z§trace_1
═B╩
#__inference__wrapped_model_10428931inputs"ў
Љ▓Ї
FullArgSpec
argsџ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
"
	optimizer
 "
trackable_dict_wrapper
-
■serving_default"
signature_map
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
 non_trainable_variables
ђlayers
Ђmetrics
 ѓlayer_regularization_losses
Ѓlayer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
ч
ёtrace_02▄
5__inference_heuristic_detector_layer_call_fn_10432568б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zёtrace_0
ќ
Ёtrace_02э
P__inference_heuristic_detector_layer_call_and_return_conditional_losses_10432579б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЁtrace_0
4:2│2heuristic_detector/kernel
&:$│2heuristic_detector/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
.
s0
t1"
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
єnon_trainable_variables
Єlayers
ѕmetrics
 Ѕlayer_regularization_losses
іlayer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
э
Іtrace_02п
1__inference_expand_1_11x11_layer_call_fn_10432588б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zІtrace_0
њ
їtrace_02з
L__inference_expand_1_11x11_layer_call_and_return_conditional_losses_10432599б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zїtrace_0
0:.ђ2expand_1_11x11/kernel
": ђ2expand_1_11x11/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Їnon_trainable_variables
јlayers
Јmetrics
 љlayer_regularization_losses
Љlayer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
ч
њtrace_02▄
5__inference_heuristic_priority_layer_call_fn_10432608б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zњtrace_0
ќ
Њtrace_02э
P__inference_heuristic_priority_layer_call_and_return_conditional_losses_10432619б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЊtrace_0
4:2│2heuristic_priority/kernel
%:#2heuristic_priority/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
0
Ё0
є1"
trackable_list_wrapper
0
Ё0
є1"
trackable_list_wrapper
 "
trackable_list_wrapper
и
ћnon_trainable_variables
Ћlayers
ќmetrics
 Ќlayer_regularization_losses
ўlayer_metrics
	variables
ђtrainable_variables
Ђregularization_losses
Ѓ__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
_generic_user_object
э
Ўtrace_02п
1__inference_contract_1_5x5_layer_call_fn_10432628б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЎtrace_0
њ
џtrace_02з
L__inference_contract_1_5x5_layer_call_and_return_conditional_losses_10432639б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zџtrace_0
0:.ђ2contract_1_5x5/kernel
!:2contract_1_5x5/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Џnon_trainable_variables
юlayers
Юmetrics
 ъlayer_regularization_losses
Ъlayer_metrics
ѕ	variables
Ѕtrainable_variables
іregularization_losses
ї__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses"
_generic_user_object
э
аtrace_02п
1__inference_concatenate_19_layer_call_fn_10432645б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zаtrace_0
њ
Аtrace_02з
L__inference_concatenate_19_layer_call_and_return_conditional_losses_10432652б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zАtrace_0
0
ћ0
Ћ1"
trackable_list_wrapper
0
ћ0
Ћ1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
бnon_trainable_variables
Бlayers
цmetrics
 Цlayer_regularization_losses
дlayer_metrics
ј	variables
Јtrainable_variables
љregularization_losses
њ__call__
+Њ&call_and_return_all_conditional_losses
'Њ"call_and_return_conditional_losses"
_generic_user_object
ш
Дtrace_02о
/__inference_expand_2_5x5_layer_call_fn_10432661б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zДtrace_0
љ
еtrace_02ы
J__inference_expand_2_5x5_layer_call_and_return_conditional_losses_10432672б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zеtrace_0
-:+	 2expand_2_5x5/kernel
: 2expand_2_5x5/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
0
Ю0
ъ1"
trackable_list_wrapper
0
Ю0
ъ1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Еnon_trainable_variables
фlayers
Фmetrics
 гlayer_regularization_losses
Гlayer_metrics
Ќ	variables
ўtrainable_variables
Ўregularization_losses
Џ__call__
+ю&call_and_return_all_conditional_losses
'ю"call_and_return_conditional_losses"
_generic_user_object
э
«trace_02п
1__inference_contract_2_3x3_layer_call_fn_10432681б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z«trace_0
њ
»trace_02з
L__inference_contract_2_3x3_layer_call_and_return_conditional_losses_10432692б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z»trace_0
/:- 2contract_2_3x3/kernel
!:2contract_2_3x3/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
░non_trainable_variables
▒layers
▓metrics
 │layer_regularization_losses
┤layer_metrics
а	variables
Аtrainable_variables
бregularization_losses
ц__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
№
хtrace_02л
)__inference_skip_2_layer_call_fn_10432698б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zхtrace_0
і
Хtrace_02в
D__inference_skip_2_layer_call_and_return_conditional_losses_10432704б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zХtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
иnon_trainable_variables
Иlayers
╣metrics
 ║layer_regularization_losses
╗layer_metrics
д	variables
Дtrainable_variables
еregularization_losses
ф__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
э
╝trace_02п
1__inference_concatenate_20_layer_call_fn_10432710б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╝trace_0
њ
йtrace_02з
L__inference_concatenate_20_layer_call_and_return_conditional_losses_10432717б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zйtrace_0
0
▓0
│1"
trackable_list_wrapper
0
▓0
│1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Йnon_trainable_variables
┐layers
└metrics
 ┴layer_regularization_losses
┬layer_metrics
г	variables
Гtrainable_variables
«regularization_losses
░__call__
+▒&call_and_return_all_conditional_losses
'▒"call_and_return_conditional_losses"
_generic_user_object
ш
├trace_02о
/__inference_expand_3_5x5_layer_call_fn_10432726б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z├trace_0
љ
─trace_02ы
J__inference_expand_3_5x5_layer_call_and_return_conditional_losses_10432737б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z─trace_0
-:+	 2expand_3_5x5/kernel
: 2expand_3_5x5/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
0
╗0
╝1"
trackable_list_wrapper
0
╗0
╝1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
┼non_trainable_variables
кlayers
Кmetrics
 ╚layer_regularization_losses
╔layer_metrics
х	variables
Хtrainable_variables
иregularization_losses
╣__call__
+║&call_and_return_all_conditional_losses
'║"call_and_return_conditional_losses"
_generic_user_object
э
╩trace_02п
1__inference_contract_3_3x3_layer_call_fn_10432746б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╩trace_0
њ
╦trace_02з
L__inference_contract_3_3x3_layer_call_and_return_conditional_losses_10432757б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╦trace_0
/:- 2contract_3_3x3/kernel
!:2contract_3_3x3/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
╠non_trainable_variables
═layers
╬metrics
 ¤layer_regularization_losses
лlayer_metrics
Й	variables
┐trainable_variables
└regularization_losses
┬__call__
+├&call_and_return_all_conditional_losses
'├"call_and_return_conditional_losses"
_generic_user_object
№
Лtrace_02л
)__inference_skip_3_layer_call_fn_10432763б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЛtrace_0
і
мtrace_02в
D__inference_skip_3_layer_call_and_return_conditional_losses_10432769б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zмtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Мnon_trainable_variables
нlayers
Нmetrics
 оlayer_regularization_losses
Оlayer_metrics
─	variables
┼trainable_variables
кregularization_losses
╚__call__
+╔&call_and_return_all_conditional_losses
'╔"call_and_return_conditional_losses"
_generic_user_object
э
пtrace_02п
1__inference_concatenate_21_layer_call_fn_10432775б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zпtrace_0
њ
┘trace_02з
L__inference_concatenate_21_layer_call_and_return_conditional_losses_10432782б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z┘trace_0
0
л0
Л1"
trackable_list_wrapper
0
л0
Л1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
┌non_trainable_variables
█layers
▄metrics
 Пlayer_regularization_losses
яlayer_metrics
╩	variables
╦trainable_variables
╠regularization_losses
╬__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses"
_generic_user_object
ш
▀trace_02о
/__inference_expand_4_5x5_layer_call_fn_10432791б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z▀trace_0
љ
Яtrace_02ы
J__inference_expand_4_5x5_layer_call_and_return_conditional_losses_10432802б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЯtrace_0
-:+	 2expand_4_5x5/kernel
: 2expand_4_5x5/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
0
┘0
┌1"
trackable_list_wrapper
0
┘0
┌1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
рnon_trainable_variables
Рlayers
сmetrics
 Сlayer_regularization_losses
тlayer_metrics
М	variables
нtrainable_variables
Нregularization_losses
О__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
э
Тtrace_02п
1__inference_contract_4_3x3_layer_call_fn_10432811б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zТtrace_0
њ
уtrace_02з
L__inference_contract_4_3x3_layer_call_and_return_conditional_losses_10432822б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zуtrace_0
/:- 2contract_4_3x3/kernel
!:2contract_4_3x3/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Уnon_trainable_variables
жlayers
Жmetrics
 вlayer_regularization_losses
Вlayer_metrics
▄	variables
Пtrainable_variables
яregularization_losses
Я__call__
+р&call_and_return_all_conditional_losses
'р"call_and_return_conditional_losses"
_generic_user_object
№
ьtrace_02л
)__inference_skip_4_layer_call_fn_10432828б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zьtrace_0
і
Ьtrace_02в
D__inference_skip_4_layer_call_and_return_conditional_losses_10432834б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЬtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
№non_trainable_variables
­layers
ыmetrics
 Ыlayer_regularization_losses
зlayer_metrics
Р	variables
сtrainable_variables
Сregularization_losses
Т__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses"
_generic_user_object
э
Зtrace_02п
1__inference_concatenate_22_layer_call_fn_10432840б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЗtrace_0
њ
шtrace_02з
L__inference_concatenate_22_layer_call_and_return_conditional_losses_10432847б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zшtrace_0
0
Ь0
№1"
trackable_list_wrapper
0
Ь0
№1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Шnon_trainable_variables
эlayers
Эmetrics
 щlayer_regularization_losses
Щlayer_metrics
У	variables
жtrainable_variables
Жregularization_losses
В__call__
+ь&call_and_return_all_conditional_losses
'ь"call_and_return_conditional_losses"
_generic_user_object
ш
чtrace_02о
/__inference_expand_5_5x5_layer_call_fn_10432856б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zчtrace_0
љ
Чtrace_02ы
J__inference_expand_5_5x5_layer_call_and_return_conditional_losses_10432867б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЧtrace_0
-:+	 2expand_5_5x5/kernel
: 2expand_5_5x5/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
0
э0
Э1"
trackable_list_wrapper
0
э0
Э1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
§non_trainable_variables
■layers
 metrics
 ђlayer_regularization_losses
Ђlayer_metrics
ы	variables
Ыtrainable_variables
зregularization_losses
ш__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
_generic_user_object
э
ѓtrace_02п
1__inference_contract_5_3x3_layer_call_fn_10432876б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zѓtrace_0
њ
Ѓtrace_02з
L__inference_contract_5_3x3_layer_call_and_return_conditional_losses_10432887б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЃtrace_0
/:- 2contract_5_3x3/kernel
!:2contract_5_3x3/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
ёnon_trainable_variables
Ёlayers
єmetrics
 Єlayer_regularization_losses
ѕlayer_metrics
Щ	variables
чtrainable_variables
Чregularization_losses
■__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
№
Ѕtrace_02л
)__inference_skip_5_layer_call_fn_10432893б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЅtrace_0
і
іtrace_02в
D__inference_skip_5_layer_call_and_return_conditional_losses_10432899б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zіtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Іnon_trainable_variables
їlayers
Їmetrics
 јlayer_regularization_losses
Јlayer_metrics
ђ	variables
Ђtrainable_variables
ѓregularization_losses
ё__call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
_generic_user_object
э
љtrace_02п
1__inference_concatenate_23_layer_call_fn_10432905б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zљtrace_0
њ
Љtrace_02з
L__inference_concatenate_23_layer_call_and_return_conditional_losses_10432912б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЉtrace_0
0
ї0
Ї1"
trackable_list_wrapper
0
ї0
Ї1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
њnon_trainable_variables
Њlayers
ћmetrics
 Ћlayer_regularization_losses
ќlayer_metrics
є	variables
Єtrainable_variables
ѕregularization_losses
і__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses"
_generic_user_object
ш
Ќtrace_02о
/__inference_expand_6_5x5_layer_call_fn_10432921б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЌtrace_0
љ
ўtrace_02ы
J__inference_expand_6_5x5_layer_call_and_return_conditional_losses_10432932б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zўtrace_0
-:+	 2expand_6_5x5/kernel
: 2expand_6_5x5/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
0
Ћ0
ќ1"
trackable_list_wrapper
0
Ћ0
ќ1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ўnon_trainable_variables
џlayers
Џmetrics
 юlayer_regularization_losses
Юlayer_metrics
Ј	variables
љtrainable_variables
Љregularization_losses
Њ__call__
+ћ&call_and_return_all_conditional_losses
'ћ"call_and_return_conditional_losses"
_generic_user_object
э
ъtrace_02п
1__inference_contract_6_3x3_layer_call_fn_10432941б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zъtrace_0
њ
Ъtrace_02з
L__inference_contract_6_3x3_layer_call_and_return_conditional_losses_10432952б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЪtrace_0
/:- 2contract_6_3x3/kernel
!:2contract_6_3x3/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
аnon_trainable_variables
Аlayers
бmetrics
 Бlayer_regularization_losses
цlayer_metrics
ў	variables
Ўtrainable_variables
џregularization_losses
ю__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
№
Цtrace_02л
)__inference_skip_6_layer_call_fn_10432958б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЦtrace_0
і
дtrace_02в
D__inference_skip_6_layer_call_and_return_conditional_losses_10432964б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zдtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Дnon_trainable_variables
еlayers
Еmetrics
 фlayer_regularization_losses
Фlayer_metrics
ъ	variables
Ъtrainable_variables
аregularization_losses
б__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
э
гtrace_02п
1__inference_concatenate_24_layer_call_fn_10432970б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zгtrace_0
њ
Гtrace_02з
L__inference_concatenate_24_layer_call_and_return_conditional_losses_10432977б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zГtrace_0
0
ф0
Ф1"
trackable_list_wrapper
0
ф0
Ф1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
«non_trainable_variables
»layers
░metrics
 ▒layer_regularization_losses
▓layer_metrics
ц	variables
Цtrainable_variables
дregularization_losses
е__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
ш
│trace_02о
/__inference_expand_7_5x5_layer_call_fn_10432986б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z│trace_0
љ
┤trace_02ы
J__inference_expand_7_5x5_layer_call_and_return_conditional_losses_10432997б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z┤trace_0
-:+	 2expand_7_5x5/kernel
: 2expand_7_5x5/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
0
│0
┤1"
trackable_list_wrapper
0
│0
┤1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
хnon_trainable_variables
Хlayers
иmetrics
 Иlayer_regularization_losses
╣layer_metrics
Г	variables
«trainable_variables
»regularization_losses
▒__call__
+▓&call_and_return_all_conditional_losses
'▓"call_and_return_conditional_losses"
_generic_user_object
э
║trace_02п
1__inference_contract_7_3x3_layer_call_fn_10433006б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z║trace_0
њ
╗trace_02з
L__inference_contract_7_3x3_layer_call_and_return_conditional_losses_10433017б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╗trace_0
/:- 2contract_7_3x3/kernel
!:2contract_7_3x3/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
╝non_trainable_variables
йlayers
Йmetrics
 ┐layer_regularization_losses
└layer_metrics
Х	variables
иtrainable_variables
Иregularization_losses
║__call__
+╗&call_and_return_all_conditional_losses
'╗"call_and_return_conditional_losses"
_generic_user_object
№
┴trace_02л
)__inference_skip_7_layer_call_fn_10433023б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z┴trace_0
і
┬trace_02в
D__inference_skip_7_layer_call_and_return_conditional_losses_10433029б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z┬trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
├non_trainable_variables
─layers
┼metrics
 кlayer_regularization_losses
Кlayer_metrics
╝	variables
йtrainable_variables
Йregularization_losses
└__call__
+┴&call_and_return_all_conditional_losses
'┴"call_and_return_conditional_losses"
_generic_user_object
э
╚trace_02п
1__inference_concatenate_25_layer_call_fn_10433035б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╚trace_0
њ
╔trace_02з
L__inference_concatenate_25_layer_call_and_return_conditional_losses_10433042б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╔trace_0
0
╚0
╔1"
trackable_list_wrapper
0
╚0
╔1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
╩non_trainable_variables
╦layers
╠metrics
 ═layer_regularization_losses
╬layer_metrics
┬	variables
├trainable_variables
─regularization_losses
к__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
ш
¤trace_02о
/__inference_expand_8_5x5_layer_call_fn_10433051б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z¤trace_0
љ
лtrace_02ы
J__inference_expand_8_5x5_layer_call_and_return_conditional_losses_10433062б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zлtrace_0
-:+	 2expand_8_5x5/kernel
: 2expand_8_5x5/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
0
Л0
м1"
trackable_list_wrapper
0
Л0
м1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Лnon_trainable_variables
мlayers
Мmetrics
 нlayer_regularization_losses
Нlayer_metrics
╦	variables
╠trainable_variables
═regularization_losses
¤__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
_generic_user_object
э
оtrace_02п
1__inference_contract_8_3x3_layer_call_fn_10433071б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zоtrace_0
њ
Оtrace_02з
L__inference_contract_8_3x3_layer_call_and_return_conditional_losses_10433082б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zОtrace_0
/:- 2contract_8_3x3/kernel
!:2contract_8_3x3/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
пnon_trainable_variables
┘layers
┌metrics
 █layer_regularization_losses
▄layer_metrics
н	variables
Нtrainable_variables
оregularization_losses
п__call__
+┘&call_and_return_all_conditional_losses
'┘"call_and_return_conditional_losses"
_generic_user_object
№
Пtrace_02л
)__inference_skip_8_layer_call_fn_10433088б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zПtrace_0
і
яtrace_02в
D__inference_skip_8_layer_call_and_return_conditional_losses_10433094б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zяtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
▀non_trainable_variables
Яlayers
рmetrics
 Рlayer_regularization_losses
сlayer_metrics
┌	variables
█trainable_variables
▄regularization_losses
я__call__
+▀&call_and_return_all_conditional_losses
'▀"call_and_return_conditional_losses"
_generic_user_object
э
Сtrace_02п
1__inference_concatenate_26_layer_call_fn_10433100б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zСtrace_0
њ
тtrace_02з
L__inference_concatenate_26_layer_call_and_return_conditional_losses_10433107б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zтtrace_0
0
Т0
у1"
trackable_list_wrapper
0
Т0
у1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Тnon_trainable_variables
уlayers
Уmetrics
 жlayer_regularization_losses
Жlayer_metrics
Я	variables
рtrainable_variables
Рregularization_losses
С__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
ш
вtrace_02о
/__inference_expand_9_5x5_layer_call_fn_10433116б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zвtrace_0
љ
Вtrace_02ы
J__inference_expand_9_5x5_layer_call_and_return_conditional_losses_10433127б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zВtrace_0
-:+	 2expand_9_5x5/kernel
: 2expand_9_5x5/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
0
№0
­1"
trackable_list_wrapper
0
№0
­1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
ьnon_trainable_variables
Ьlayers
№metrics
 ­layer_regularization_losses
ыlayer_metrics
ж	variables
Жtrainable_variables
вregularization_losses
ь__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
э
Ыtrace_02п
1__inference_contract_9_3x3_layer_call_fn_10433136б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЫtrace_0
њ
зtrace_02з
L__inference_contract_9_3x3_layer_call_and_return_conditional_losses_10433147б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zзtrace_0
/:- 2contract_9_3x3/kernel
!:2contract_9_3x3/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Зnon_trainable_variables
шlayers
Шmetrics
 эlayer_regularization_losses
Эlayer_metrics
Ы	variables
зtrainable_variables
Зregularization_losses
Ш__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses"
_generic_user_object
№
щtrace_02л
)__inference_skip_9_layer_call_fn_10433153б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zщtrace_0
і
Щtrace_02в
D__inference_skip_9_layer_call_and_return_conditional_losses_10433159б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЩtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
чnon_trainable_variables
Чlayers
§metrics
 ■layer_regularization_losses
 layer_metrics
Э	variables
щtrainable_variables
Щregularization_losses
Ч__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
э
ђtrace_02п
1__inference_concatenate_27_layer_call_fn_10433165б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zђtrace_0
њ
Ђtrace_02з
L__inference_concatenate_27_layer_call_and_return_conditional_losses_10433172б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЂtrace_0
0
ё0
Ё1"
trackable_list_wrapper
0
ё0
Ё1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
ѓnon_trainable_variables
Ѓlayers
ёmetrics
 Ёlayer_regularization_losses
єlayer_metrics
■	variables
 trainable_variables
ђregularization_losses
ѓ__call__
+Ѓ&call_and_return_all_conditional_losses
'Ѓ"call_and_return_conditional_losses"
_generic_user_object
Ш
Єtrace_02О
0__inference_expand_10_5x5_layer_call_fn_10433181б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЄtrace_0
Љ
ѕtrace_02Ы
K__inference_expand_10_5x5_layer_call_and_return_conditional_losses_10433192б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zѕtrace_0
.:,	 2expand_10_5x5/kernel
 : 2expand_10_5x5/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
0
Ї0
ј1"
trackable_list_wrapper
0
Ї0
ј1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ѕnon_trainable_variables
іlayers
Іmetrics
 їlayer_regularization_losses
Їlayer_metrics
Є	variables
ѕtrainable_variables
Ѕregularization_losses
І__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses"
_generic_user_object
Э
јtrace_02┘
2__inference_contract_10_3x3_layer_call_fn_10433201б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zјtrace_0
Њ
Јtrace_02З
M__inference_contract_10_3x3_layer_call_and_return_conditional_losses_10433212б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЈtrace_0
0:. 2contract_10_3x3/kernel
": 2contract_10_3x3/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
љnon_trainable_variables
Љlayers
њmetrics
 Њlayer_regularization_losses
ћlayer_metrics
љ	variables
Љtrainable_variables
њregularization_losses
ћ__call__
+Ћ&call_and_return_all_conditional_losses
'Ћ"call_and_return_conditional_losses"
_generic_user_object
­
Ћtrace_02Л
*__inference_skip_10_layer_call_fn_10433218б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЋtrace_0
І
ќtrace_02В
E__inference_skip_10_layer_call_and_return_conditional_losses_10433224б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zќtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ќnon_trainable_variables
ўlayers
Ўmetrics
 џlayer_regularization_losses
Џlayer_metrics
ќ	variables
Ќtrainable_variables
ўregularization_losses
џ__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses"
_generic_user_object
э
юtrace_02п
1__inference_concatenate_28_layer_call_fn_10433230б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zюtrace_0
њ
Юtrace_02з
L__inference_concatenate_28_layer_call_and_return_conditional_losses_10433237б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЮtrace_0
0
б0
Б1"
trackable_list_wrapper
0
б0
Б1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
ъnon_trainable_variables
Ъlayers
аmetrics
 Аlayer_regularization_losses
бlayer_metrics
ю	variables
Юtrainable_variables
ъregularization_losses
а__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
Ш
Бtrace_02О
0__inference_expand_11_5x5_layer_call_fn_10433246б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zБtrace_0
Љ
цtrace_02Ы
K__inference_expand_11_5x5_layer_call_and_return_conditional_losses_10433257б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zцtrace_0
.:,	 2expand_11_5x5/kernel
 : 2expand_11_5x5/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
0
Ф0
г1"
trackable_list_wrapper
0
Ф0
г1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Цnon_trainable_variables
дlayers
Дmetrics
 еlayer_regularization_losses
Еlayer_metrics
Ц	variables
дtrainable_variables
Дregularization_losses
Е__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
_generic_user_object
Э
фtrace_02┘
2__inference_contract_11_3x3_layer_call_fn_10433266б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zфtrace_0
Њ
Фtrace_02З
M__inference_contract_11_3x3_layer_call_and_return_conditional_losses_10433277б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zФtrace_0
0:. 2contract_11_3x3/kernel
": 2contract_11_3x3/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
гnon_trainable_variables
Гlayers
«metrics
 »layer_regularization_losses
░layer_metrics
«	variables
»trainable_variables
░regularization_losses
▓__call__
+│&call_and_return_all_conditional_losses
'│"call_and_return_conditional_losses"
_generic_user_object
­
▒trace_02Л
*__inference_skip_11_layer_call_fn_10433283б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z▒trace_0
І
▓trace_02В
E__inference_skip_11_layer_call_and_return_conditional_losses_10433289б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z▓trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
│non_trainable_variables
┤layers
хmetrics
 Хlayer_regularization_losses
иlayer_metrics
┤	variables
хtrainable_variables
Хregularization_losses
И__call__
+╣&call_and_return_all_conditional_losses
'╣"call_and_return_conditional_losses"
_generic_user_object
э
Иtrace_02п
1__inference_concatenate_29_layer_call_fn_10433295б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zИtrace_0
њ
╣trace_02з
L__inference_concatenate_29_layer_call_and_return_conditional_losses_10433302б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╣trace_0
0
└0
┴1"
trackable_list_wrapper
0
└0
┴1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
║non_trainable_variables
╗layers
╝metrics
 йlayer_regularization_losses
Йlayer_metrics
║	variables
╗trainable_variables
╝regularization_losses
Й__call__
+┐&call_and_return_all_conditional_losses
'┐"call_and_return_conditional_losses"
_generic_user_object
Ш
┐trace_02О
0__inference_expand_12_5x5_layer_call_fn_10433311б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z┐trace_0
Љ
└trace_02Ы
K__inference_expand_12_5x5_layer_call_and_return_conditional_losses_10433322б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z└trace_0
.:,	 2expand_12_5x5/kernel
 : 2expand_12_5x5/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
0
╔0
╩1"
trackable_list_wrapper
0
╔0
╩1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
┴non_trainable_variables
┬layers
├metrics
 ─layer_regularization_losses
┼layer_metrics
├	variables
─trainable_variables
┼regularization_losses
К__call__
+╚&call_and_return_all_conditional_losses
'╚"call_and_return_conditional_losses"
_generic_user_object
Э
кtrace_02┘
2__inference_contract_12_3x3_layer_call_fn_10433331б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zкtrace_0
Њ
Кtrace_02З
M__inference_contract_12_3x3_layer_call_and_return_conditional_losses_10433342б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zКtrace_0
0:. 2contract_12_3x3/kernel
": 2contract_12_3x3/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
╚non_trainable_variables
╔layers
╩metrics
 ╦layer_regularization_losses
╠layer_metrics
╠	variables
═trainable_variables
╬regularization_losses
л__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
­
═trace_02Л
*__inference_skip_12_layer_call_fn_10433348б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z═trace_0
І
╬trace_02В
E__inference_skip_12_layer_call_and_return_conditional_losses_10433354б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╬trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
¤non_trainable_variables
лlayers
Лmetrics
 мlayer_regularization_losses
Мlayer_metrics
м	variables
Мtrainable_variables
нregularization_losses
о__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
э
нtrace_02п
1__inference_concatenate_30_layer_call_fn_10433360б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zнtrace_0
њ
Нtrace_02з
L__inference_concatenate_30_layer_call_and_return_conditional_losses_10433367б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zНtrace_0
0
я0
▀1"
trackable_list_wrapper
0
я0
▀1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
оnon_trainable_variables
Оlayers
пmetrics
 ┘layer_regularization_losses
┌layer_metrics
п	variables
┘trainable_variables
┌regularization_losses
▄__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
Ш
█trace_02О
0__inference_expand_13_5x5_layer_call_fn_10433376б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z█trace_0
Љ
▄trace_02Ы
K__inference_expand_13_5x5_layer_call_and_return_conditional_losses_10433387б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z▄trace_0
.:,	 2expand_13_5x5/kernel
 : 2expand_13_5x5/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
0
у0
У1"
trackable_list_wrapper
0
у0
У1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Пnon_trainable_variables
яlayers
▀metrics
 Яlayer_regularization_losses
рlayer_metrics
р	variables
Рtrainable_variables
сregularization_losses
т__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
_generic_user_object
Э
Рtrace_02┘
2__inference_contract_13_3x3_layer_call_fn_10433396б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zРtrace_0
Њ
сtrace_02З
M__inference_contract_13_3x3_layer_call_and_return_conditional_losses_10433407б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zсtrace_0
0:. 2contract_13_3x3/kernel
": 2contract_13_3x3/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Сnon_trainable_variables
тlayers
Тmetrics
 уlayer_regularization_losses
Уlayer_metrics
Ж	variables
вtrainable_variables
Вregularization_losses
Ь__call__
+№&call_and_return_all_conditional_losses
'№"call_and_return_conditional_losses"
_generic_user_object
­
жtrace_02Л
*__inference_skip_13_layer_call_fn_10433413б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zжtrace_0
І
Жtrace_02В
E__inference_skip_13_layer_call_and_return_conditional_losses_10433419б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЖtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
вnon_trainable_variables
Вlayers
ьmetrics
 Ьlayer_regularization_losses
№layer_metrics
­	variables
ыtrainable_variables
Ыregularization_losses
З__call__
+ш&call_and_return_all_conditional_losses
'ш"call_and_return_conditional_losses"
_generic_user_object
э
­trace_02п
1__inference_concatenate_31_layer_call_fn_10433425б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z­trace_0
њ
ыtrace_02з
L__inference_concatenate_31_layer_call_and_return_conditional_losses_10433432б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zыtrace_0
0
Ч0
§1"
trackable_list_wrapper
0
Ч0
§1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ыnon_trainable_variables
зlayers
Зmetrics
 шlayer_regularization_losses
Шlayer_metrics
Ш	variables
эtrainable_variables
Эregularization_losses
Щ__call__
+ч&call_and_return_all_conditional_losses
'ч"call_and_return_conditional_losses"
_generic_user_object
Ш
эtrace_02О
0__inference_expand_14_5x5_layer_call_fn_10433441б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zэtrace_0
Љ
Эtrace_02Ы
K__inference_expand_14_5x5_layer_call_and_return_conditional_losses_10433452б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЭtrace_0
.:,	 2expand_14_5x5/kernel
 : 2expand_14_5x5/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
0
Ё0
є1"
trackable_list_wrapper
0
Ё0
є1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
щnon_trainable_variables
Щlayers
чmetrics
 Чlayer_regularization_losses
§layer_metrics
 	variables
ђtrainable_variables
Ђregularization_losses
Ѓ__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
_generic_user_object
Э
■trace_02┘
2__inference_contract_14_3x3_layer_call_fn_10433461б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z■trace_0
Њ
 trace_02З
M__inference_contract_14_3x3_layer_call_and_return_conditional_losses_10433472б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z trace_0
0:. 2contract_14_3x3/kernel
": 2contract_14_3x3/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
ђ	non_trainable_variables
Ђ	layers
ѓ	metrics
 Ѓ	layer_regularization_losses
ё	layer_metrics
ѕ	variables
Ѕtrainable_variables
іregularization_losses
ї__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses"
_generic_user_object
­
Ё	trace_02Л
*__inference_skip_14_layer_call_fn_10433478б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЁ	trace_0
І
є	trace_02В
E__inference_skip_14_layer_call_and_return_conditional_losses_10433484б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zє	trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Є	non_trainable_variables
ѕ	layers
Ѕ	metrics
 і	layer_regularization_losses
І	layer_metrics
ј	variables
Јtrainable_variables
љregularization_losses
њ__call__
+Њ&call_and_return_all_conditional_losses
'Њ"call_and_return_conditional_losses"
_generic_user_object
э
ї	trace_02п
1__inference_concatenate_32_layer_call_fn_10433490б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zї	trace_0
њ
Ї	trace_02з
L__inference_concatenate_32_layer_call_and_return_conditional_losses_10433497б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЇ	trace_0
0
џ0
Џ1"
trackable_list_wrapper
0
џ0
Џ1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
ј	non_trainable_variables
Ј	layers
љ	metrics
 Љ	layer_regularization_losses
њ	layer_metrics
ћ	variables
Ћtrainable_variables
ќregularization_losses
ў__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
_generic_user_object
Ш
Њ	trace_02О
0__inference_expand_15_5x5_layer_call_fn_10433506б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЊ	trace_0
Љ
ћ	trace_02Ы
K__inference_expand_15_5x5_layer_call_and_return_conditional_losses_10433517б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zћ	trace_0
.:,	 2expand_15_5x5/kernel
 : 2expand_15_5x5/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
0
Б0
ц1"
trackable_list_wrapper
0
Б0
ц1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ћ	non_trainable_variables
ќ	layers
Ќ	metrics
 ў	layer_regularization_losses
Ў	layer_metrics
Ю	variables
ъtrainable_variables
Ъregularization_losses
А__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
Э
џ	trace_02┘
2__inference_contract_15_3x3_layer_call_fn_10433526б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zџ	trace_0
Њ
Џ	trace_02З
M__inference_contract_15_3x3_layer_call_and_return_conditional_losses_10433537б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЏ	trace_0
0:. 2contract_15_3x3/kernel
": 2contract_15_3x3/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
ю	non_trainable_variables
Ю	layers
ъ	metrics
 Ъ	layer_regularization_losses
а	layer_metrics
д	variables
Дtrainable_variables
еregularization_losses
ф__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
­
А	trace_02Л
*__inference_skip_15_layer_call_fn_10433543б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zА	trace_0
І
б	trace_02В
E__inference_skip_15_layer_call_and_return_conditional_losses_10433549б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zб	trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Б	non_trainable_variables
ц	layers
Ц	metrics
 д	layer_regularization_losses
Д	layer_metrics
г	variables
Гtrainable_variables
«regularization_losses
░__call__
+▒&call_and_return_all_conditional_losses
'▒"call_and_return_conditional_losses"
_generic_user_object
э
е	trace_02п
1__inference_concatenate_33_layer_call_fn_10433555б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zе	trace_0
њ
Е	trace_02з
L__inference_concatenate_33_layer_call_and_return_conditional_losses_10433562б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЕ	trace_0
0
И0
╣1"
trackable_list_wrapper
0
И0
╣1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
ф	non_trainable_variables
Ф	layers
г	metrics
 Г	layer_regularization_losses
«	layer_metrics
▓	variables
│trainable_variables
┤regularization_losses
Х__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses"
_generic_user_object
Ш
»	trace_02О
0__inference_expand_16_5x5_layer_call_fn_10433571б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z»	trace_0
Љ
░	trace_02Ы
K__inference_expand_16_5x5_layer_call_and_return_conditional_losses_10433582б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z░	trace_0
.:,	 2expand_16_5x5/kernel
 : 2expand_16_5x5/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
0
┴0
┬1"
trackable_list_wrapper
0
┴0
┬1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
▒	non_trainable_variables
▓	layers
│	metrics
 ┤	layer_regularization_losses
х	layer_metrics
╗	variables
╝trainable_variables
йregularization_losses
┐__call__
+└&call_and_return_all_conditional_losses
'└"call_and_return_conditional_losses"
_generic_user_object
Э
Х	trace_02┘
2__inference_contract_16_3x3_layer_call_fn_10433591б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zХ	trace_0
Њ
и	trace_02З
M__inference_contract_16_3x3_layer_call_and_return_conditional_losses_10433602б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zи	trace_0
0:. 2contract_16_3x3/kernel
": 2contract_16_3x3/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
И	non_trainable_variables
╣	layers
║	metrics
 ╗	layer_regularization_losses
╝	layer_metrics
─	variables
┼trainable_variables
кregularization_losses
╚__call__
+╔&call_and_return_all_conditional_losses
'╔"call_and_return_conditional_losses"
_generic_user_object
­
й	trace_02Л
*__inference_skip_16_layer_call_fn_10433608б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zй	trace_0
І
Й	trace_02В
E__inference_skip_16_layer_call_and_return_conditional_losses_10433614б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЙ	trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
┐	non_trainable_variables
└	layers
┴	metrics
 ┬	layer_regularization_losses
├	layer_metrics
╩	variables
╦trainable_variables
╠regularization_losses
╬__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses"
_generic_user_object
э
─	trace_02п
1__inference_concatenate_34_layer_call_fn_10433620б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z─	trace_0
њ
┼	trace_02з
L__inference_concatenate_34_layer_call_and_return_conditional_losses_10433627б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z┼	trace_0
0
о0
О1"
trackable_list_wrapper
0
о0
О1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
к	non_trainable_variables
К	layers
╚	metrics
 ╔	layer_regularization_losses
╩	layer_metrics
л	variables
Лtrainable_variables
мregularization_losses
н__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object
Ш
╦	trace_02О
0__inference_expand_17_5x5_layer_call_fn_10433636б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╦	trace_0
Љ
╠	trace_02Ы
K__inference_expand_17_5x5_layer_call_and_return_conditional_losses_10433647б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╠	trace_0
.:,	 2expand_17_5x5/kernel
 : 2expand_17_5x5/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
0
▀0
Я1"
trackable_list_wrapper
0
▀0
Я1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
═	non_trainable_variables
╬	layers
¤	metrics
 л	layer_regularization_losses
Л	layer_metrics
┘	variables
┌trainable_variables
█regularization_losses
П__call__
+я&call_and_return_all_conditional_losses
'я"call_and_return_conditional_losses"
_generic_user_object
Э
м	trace_02┘
2__inference_contract_17_3x3_layer_call_fn_10433656б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zм	trace_0
Њ
М	trace_02З
M__inference_contract_17_3x3_layer_call_and_return_conditional_losses_10433667б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zМ	trace_0
0:. 2contract_17_3x3/kernel
": 2contract_17_3x3/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
н	non_trainable_variables
Н	layers
о	metrics
 О	layer_regularization_losses
п	layer_metrics
Р	variables
сtrainable_variables
Сregularization_losses
Т__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses"
_generic_user_object
­
┘	trace_02Л
*__inference_skip_17_layer_call_fn_10433673б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z┘	trace_0
І
┌	trace_02В
E__inference_skip_17_layer_call_and_return_conditional_losses_10433679б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z┌	trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
█	non_trainable_variables
▄	layers
П	metrics
 я	layer_regularization_losses
▀	layer_metrics
У	variables
жtrainable_variables
Жregularization_losses
В__call__
+ь&call_and_return_all_conditional_losses
'ь"call_and_return_conditional_losses"
_generic_user_object
э
Я	trace_02п
1__inference_concatenate_35_layer_call_fn_10433685б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЯ	trace_0
њ
р	trace_02з
L__inference_concatenate_35_layer_call_and_return_conditional_losses_10433692б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zр	trace_0
0
З0
ш1"
trackable_list_wrapper
0
З0
ш1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Р	non_trainable_variables
с	layers
С	metrics
 т	layer_regularization_losses
Т	layer_metrics
Ь	variables
№trainable_variables
­regularization_losses
Ы__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
Ш
у	trace_02О
0__inference_expand_18_5x5_layer_call_fn_10433701б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zу	trace_0
Љ
У	trace_02Ы
K__inference_expand_18_5x5_layer_call_and_return_conditional_losses_10433712б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zУ	trace_0
.:,	 2expand_18_5x5/kernel
 : 2expand_18_5x5/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
0
§0
■1"
trackable_list_wrapper
0
§0
■1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
ж	non_trainable_variables
Ж	layers
в	metrics
 В	layer_regularization_losses
ь	layer_metrics
э	variables
Эtrainable_variables
щregularization_losses
ч__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
Э
Ь	trace_02┘
2__inference_contract_18_3x3_layer_call_fn_10433721б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЬ	trace_0
Њ
№	trace_02З
M__inference_contract_18_3x3_layer_call_and_return_conditional_losses_10433732б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z№	trace_0
0:. 2contract_18_3x3/kernel
": 2contract_18_3x3/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
­	non_trainable_variables
ы	layers
Ы	metrics
 з	layer_regularization_losses
З	layer_metrics
ђ	variables
Ђtrainable_variables
ѓregularization_losses
ё__call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
_generic_user_object
­
ш	trace_02Л
*__inference_skip_18_layer_call_fn_10433738б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zш	trace_0
І
Ш	trace_02В
E__inference_skip_18_layer_call_and_return_conditional_losses_10433744б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zШ	trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
э	non_trainable_variables
Э	layers
щ	metrics
 Щ	layer_regularization_losses
ч	layer_metrics
є	variables
Єtrainable_variables
ѕregularization_losses
і__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses"
_generic_user_object
э
Ч	trace_02п
1__inference_concatenate_36_layer_call_fn_10433750б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЧ	trace_0
њ
§	trace_02з
L__inference_concatenate_36_layer_call_and_return_conditional_losses_10433757б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z§	trace_0
0
њ0
Њ1"
trackable_list_wrapper
0
њ0
Њ1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
■	non_trainable_variables
 	layers
ђ
metrics
 Ђ
layer_regularization_losses
ѓ
layer_metrics
ї	variables
Їtrainable_variables
јregularization_losses
љ__call__
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses"
_generic_user_object
Ш
Ѓ
trace_02О
0__inference_expand_19_5x5_layer_call_fn_10433766б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЃ
trace_0
Љ
ё
trace_02Ы
K__inference_expand_19_5x5_layer_call_and_return_conditional_losses_10433777б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zё
trace_0
.:,	 2expand_19_5x5/kernel
 : 2expand_19_5x5/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
0
Џ0
ю1"
trackable_list_wrapper
0
Џ0
ю1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ё
non_trainable_variables
є
layers
Є
metrics
 ѕ
layer_regularization_losses
Ѕ
layer_metrics
Ћ	variables
ќtrainable_variables
Ќregularization_losses
Ў__call__
+џ&call_and_return_all_conditional_losses
'џ"call_and_return_conditional_losses"
_generic_user_object
Э
і
trace_02┘
2__inference_contract_19_3x3_layer_call_fn_10433786б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zі
trace_0
Њ
І
trace_02З
M__inference_contract_19_3x3_layer_call_and_return_conditional_losses_10433797б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zІ
trace_0
0:. 2contract_19_3x3/kernel
": 2contract_19_3x3/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
ї
non_trainable_variables
Ї
layers
ј
metrics
 Ј
layer_regularization_losses
љ
layer_metrics
ъ	variables
Ъtrainable_variables
аregularization_losses
б__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
­
Љ
trace_02Л
*__inference_skip_19_layer_call_fn_10433803б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЉ
trace_0
І
њ
trace_02В
E__inference_skip_19_layer_call_and_return_conditional_losses_10433809б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zњ
trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Њ
non_trainable_variables
ћ
layers
Ћ
metrics
 ќ
layer_regularization_losses
Ќ
layer_metrics
ц	variables
Цtrainable_variables
дregularization_losses
е__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
э
ў
trace_02п
1__inference_concatenate_37_layer_call_fn_10433815б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zў
trace_0
њ
Ў
trace_02з
L__inference_concatenate_37_layer_call_and_return_conditional_losses_10433822б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЎ
trace_0
0
░0
▒1"
trackable_list_wrapper
0
░0
▒1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
џ
non_trainable_variables
Џ
layers
ю
metrics
 Ю
layer_regularization_losses
ъ
layer_metrics
ф	variables
Фtrainable_variables
гregularization_losses
«__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
Ш
Ъ
trace_02О
0__inference_expand_20_5x5_layer_call_fn_10433831б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЪ
trace_0
Љ
а
trace_02Ы
K__inference_expand_20_5x5_layer_call_and_return_conditional_losses_10433842б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zа
trace_0
.:,	 2expand_20_5x5/kernel
 : 2expand_20_5x5/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
0
╣0
║1"
trackable_list_wrapper
0
╣0
║1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
А
non_trainable_variables
б
layers
Б
metrics
 ц
layer_regularization_losses
Ц
layer_metrics
│	variables
┤trainable_variables
хregularization_losses
и__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
Э
д
trace_02┘
2__inference_contract_20_3x3_layer_call_fn_10433851б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zд
trace_0
Њ
Д
trace_02З
M__inference_contract_20_3x3_layer_call_and_return_conditional_losses_10433862б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zД
trace_0
0:. 2contract_20_3x3/kernel
": 2contract_20_3x3/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
е
non_trainable_variables
Е
layers
ф
metrics
 Ф
layer_regularization_losses
г
layer_metrics
╝	variables
йtrainable_variables
Йregularization_losses
└__call__
+┴&call_and_return_all_conditional_losses
'┴"call_and_return_conditional_losses"
_generic_user_object
­
Г
trace_02Л
*__inference_skip_20_layer_call_fn_10433868б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zГ
trace_0
І
«
trace_02В
E__inference_skip_20_layer_call_and_return_conditional_losses_10433874б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z«
trace_0
0
╚0
╔1"
trackable_list_wrapper
0
╚0
╔1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
»
non_trainable_variables
░
layers
▒
metrics
 ▓
layer_regularization_losses
│
layer_metrics
┬	variables
├trainable_variables
─regularization_losses
к__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
Щ
┤
trace_02█
4__inference_policy_aggregator_layer_call_fn_10433883б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z┤
trace_0
Ћ
х
trace_02Ш
O__inference_policy_aggregator_layer_call_and_return_conditional_losses_10433894б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zх
trace_0
2:02policy_aggregator/kernel
$:"2policy_aggregator/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Х
non_trainable_variables
и
layers
И
metrics
 ╣
layer_regularization_losses
║
layer_metrics
╦	variables
╠trainable_variables
═regularization_losses
¤__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
_generic_user_object
Э
╗
trace_02┘
2__inference_all_value_input_layer_call_fn_10433900б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╗
trace_0
Њ
╝
trace_02З
M__inference_all_value_input_layer_call_and_return_conditional_losses_10433907б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╝
trace_0
0
О0
п1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
й
non_trainable_variables
Й
layers
┐
metrics
 └
layer_regularization_losses
┴
layer_metrics
Л	variables
мtrainable_variables
Мregularization_losses
Н__call__
+о&call_and_return_all_conditional_losses
'о"call_and_return_conditional_losses"
_generic_user_object
з
┬
trace_02н
-__inference_border_off_layer_call_fn_10433916б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z┬
trace_0
ј
├
trace_02№
H__inference_border_off_layer_call_and_return_conditional_losses_10433926б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z├
trace_0
+:)2border_off/kernel
:2border_off/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
─
non_trainable_variables
┼
layers
к
metrics
 К
layer_regularization_losses
╚
layer_metrics
┌	variables
█trainable_variables
▄regularization_losses
я__call__
+▀&call_and_return_all_conditional_losses
'▀"call_and_return_conditional_losses"
_generic_user_object
щ
╔
trace_02┌
3__inference_flat_value_input_layer_call_fn_10433931б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╔
trace_0
ћ
╩
trace_02ш
N__inference_flat_value_input_layer_call_and_return_conditional_losses_10433937б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╩
trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
╦
non_trainable_variables
╠
layers
═
metrics
 ╬
layer_regularization_losses
¤
layer_metrics
Я	variables
рtrainable_variables
Рregularization_losses
С__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
З
л
trace_02Н
.__inference_flat_logits_layer_call_fn_10433942б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zл
trace_0
Ј
Л
trace_02­
I__inference_flat_logits_layer_call_and_return_conditional_losses_10433948б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЛ
trace_0
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
м
non_trainable_variables
М
layers
н
metrics
 Н
layer_regularization_losses
о
layer_metrics
у	variables
Уtrainable_variables
жregularization_losses
в__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
Ђ
О
trace_02Р
.__inference_policy_head_layer_call_fn_10433953»
д▓б
FullArgSpec%
argsџ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsб

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zО
trace_0
ю
п
trace_02§
I__inference_policy_head_layer_call_and_return_conditional_losses_10433958»
д▓б
FullArgSpec%
argsџ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsб

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zп
trace_0
0
з0
З1"
trackable_list_wrapper
0
з0
З1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
┘
non_trainable_variables
┌
layers
█
metrics
 ▄
layer_regularization_losses
П
layer_metrics
ь	variables
Ьtrainable_variables
№regularization_losses
ы__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
_generic_user_object
з
я
trace_02н
-__inference_value_head_layer_call_fn_10433967б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zя
trace_0
ј
▀
trace_02№
H__inference_value_head_layer_call_and_return_conditional_losses_10433978б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z▀
trace_0
$:"	т,2value_head/kernel
:2value_head/bias
L
j0
k1
|2
}3
О4
п5"
trackable_list_wrapper
я
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44
.45
/46
047
148
249
350
451
552
653
754
855
956
:57
;58
<59
=60
>61
?62
@63
A64
B65
C66
D67
E68
F69
G70
H71
I72
J73
K74
L75
M76
N77
O78
P79
Q80
R81
S82
T83
U84
V85
W86
X87
Y88"
trackable_list_wrapper
(
Я
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ёBЂ
2__inference_gomoku_resnet_1_layer_call_fn_10430246inputs"└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ёBЂ
2__inference_gomoku_resnet_1_layer_call_fn_10431822inputs"└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ЪBю
M__inference_gomoku_resnet_1_layer_call_and_return_conditional_losses_10432096inputs"└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ЪBю
M__inference_gomoku_resnet_1_layer_call_and_return_conditional_losses_10432370inputs"└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
╠B╔
&__inference_signature_wrapper_10432559inputs"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
жBТ
5__inference_heuristic_detector_layer_call_fn_10432568inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ёBЂ
P__inference_heuristic_detector_layer_call_and_return_conditional_losses_10432579inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
тBР
1__inference_expand_1_11x11_layer_call_fn_10432588inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ђB§
L__inference_expand_1_11x11_layer_call_and_return_conditional_losses_10432599inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
жBТ
5__inference_heuristic_priority_layer_call_fn_10432608inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ёBЂ
P__inference_heuristic_priority_layer_call_and_return_conditional_losses_10432619inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
тBР
1__inference_contract_1_5x5_layer_call_fn_10432628inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ђB§
L__inference_contract_1_5x5_layer_call_and_return_conditional_losses_10432639inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ыBЬ
1__inference_concatenate_19_layer_call_fn_10432645inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
їBЅ
L__inference_concatenate_19_layer_call_and_return_conditional_losses_10432652inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
сBЯ
/__inference_expand_2_5x5_layer_call_fn_10432661inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
■Bч
J__inference_expand_2_5x5_layer_call_and_return_conditional_losses_10432672inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
тBР
1__inference_contract_2_3x3_layer_call_fn_10432681inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ђB§
L__inference_contract_2_3x3_layer_call_and_return_conditional_losses_10432692inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
жBТ
)__inference_skip_2_layer_call_fn_10432698inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ёBЂ
D__inference_skip_2_layer_call_and_return_conditional_losses_10432704inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ыBЬ
1__inference_concatenate_20_layer_call_fn_10432710inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
їBЅ
L__inference_concatenate_20_layer_call_and_return_conditional_losses_10432717inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
сBЯ
/__inference_expand_3_5x5_layer_call_fn_10432726inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
■Bч
J__inference_expand_3_5x5_layer_call_and_return_conditional_losses_10432737inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
тBР
1__inference_contract_3_3x3_layer_call_fn_10432746inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ђB§
L__inference_contract_3_3x3_layer_call_and_return_conditional_losses_10432757inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
жBТ
)__inference_skip_3_layer_call_fn_10432763inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ёBЂ
D__inference_skip_3_layer_call_and_return_conditional_losses_10432769inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ыBЬ
1__inference_concatenate_21_layer_call_fn_10432775inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
їBЅ
L__inference_concatenate_21_layer_call_and_return_conditional_losses_10432782inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
сBЯ
/__inference_expand_4_5x5_layer_call_fn_10432791inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
■Bч
J__inference_expand_4_5x5_layer_call_and_return_conditional_losses_10432802inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
тBР
1__inference_contract_4_3x3_layer_call_fn_10432811inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ђB§
L__inference_contract_4_3x3_layer_call_and_return_conditional_losses_10432822inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
жBТ
)__inference_skip_4_layer_call_fn_10432828inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ёBЂ
D__inference_skip_4_layer_call_and_return_conditional_losses_10432834inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ыBЬ
1__inference_concatenate_22_layer_call_fn_10432840inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
їBЅ
L__inference_concatenate_22_layer_call_and_return_conditional_losses_10432847inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
сBЯ
/__inference_expand_5_5x5_layer_call_fn_10432856inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
■Bч
J__inference_expand_5_5x5_layer_call_and_return_conditional_losses_10432867inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
тBР
1__inference_contract_5_3x3_layer_call_fn_10432876inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ђB§
L__inference_contract_5_3x3_layer_call_and_return_conditional_losses_10432887inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
жBТ
)__inference_skip_5_layer_call_fn_10432893inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ёBЂ
D__inference_skip_5_layer_call_and_return_conditional_losses_10432899inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ыBЬ
1__inference_concatenate_23_layer_call_fn_10432905inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
їBЅ
L__inference_concatenate_23_layer_call_and_return_conditional_losses_10432912inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
сBЯ
/__inference_expand_6_5x5_layer_call_fn_10432921inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
■Bч
J__inference_expand_6_5x5_layer_call_and_return_conditional_losses_10432932inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
тBР
1__inference_contract_6_3x3_layer_call_fn_10432941inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ђB§
L__inference_contract_6_3x3_layer_call_and_return_conditional_losses_10432952inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
жBТ
)__inference_skip_6_layer_call_fn_10432958inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ёBЂ
D__inference_skip_6_layer_call_and_return_conditional_losses_10432964inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ыBЬ
1__inference_concatenate_24_layer_call_fn_10432970inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
їBЅ
L__inference_concatenate_24_layer_call_and_return_conditional_losses_10432977inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
сBЯ
/__inference_expand_7_5x5_layer_call_fn_10432986inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
■Bч
J__inference_expand_7_5x5_layer_call_and_return_conditional_losses_10432997inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
тBР
1__inference_contract_7_3x3_layer_call_fn_10433006inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ђB§
L__inference_contract_7_3x3_layer_call_and_return_conditional_losses_10433017inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
жBТ
)__inference_skip_7_layer_call_fn_10433023inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ёBЂ
D__inference_skip_7_layer_call_and_return_conditional_losses_10433029inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ыBЬ
1__inference_concatenate_25_layer_call_fn_10433035inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
їBЅ
L__inference_concatenate_25_layer_call_and_return_conditional_losses_10433042inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
сBЯ
/__inference_expand_8_5x5_layer_call_fn_10433051inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
■Bч
J__inference_expand_8_5x5_layer_call_and_return_conditional_losses_10433062inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
тBР
1__inference_contract_8_3x3_layer_call_fn_10433071inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ђB§
L__inference_contract_8_3x3_layer_call_and_return_conditional_losses_10433082inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
жBТ
)__inference_skip_8_layer_call_fn_10433088inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ёBЂ
D__inference_skip_8_layer_call_and_return_conditional_losses_10433094inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ыBЬ
1__inference_concatenate_26_layer_call_fn_10433100inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
їBЅ
L__inference_concatenate_26_layer_call_and_return_conditional_losses_10433107inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
сBЯ
/__inference_expand_9_5x5_layer_call_fn_10433116inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
■Bч
J__inference_expand_9_5x5_layer_call_and_return_conditional_losses_10433127inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
тBР
1__inference_contract_9_3x3_layer_call_fn_10433136inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ђB§
L__inference_contract_9_3x3_layer_call_and_return_conditional_losses_10433147inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
жBТ
)__inference_skip_9_layer_call_fn_10433153inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ёBЂ
D__inference_skip_9_layer_call_and_return_conditional_losses_10433159inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ыBЬ
1__inference_concatenate_27_layer_call_fn_10433165inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
їBЅ
L__inference_concatenate_27_layer_call_and_return_conditional_losses_10433172inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
СBр
0__inference_expand_10_5x5_layer_call_fn_10433181inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 BЧ
K__inference_expand_10_5x5_layer_call_and_return_conditional_losses_10433192inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ТBс
2__inference_contract_10_3x3_layer_call_fn_10433201inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЂB■
M__inference_contract_10_3x3_layer_call_and_return_conditional_losses_10433212inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЖBу
*__inference_skip_10_layer_call_fn_10433218inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЁBѓ
E__inference_skip_10_layer_call_and_return_conditional_losses_10433224inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ыBЬ
1__inference_concatenate_28_layer_call_fn_10433230inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
їBЅ
L__inference_concatenate_28_layer_call_and_return_conditional_losses_10433237inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
СBр
0__inference_expand_11_5x5_layer_call_fn_10433246inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 BЧ
K__inference_expand_11_5x5_layer_call_and_return_conditional_losses_10433257inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ТBс
2__inference_contract_11_3x3_layer_call_fn_10433266inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЂB■
M__inference_contract_11_3x3_layer_call_and_return_conditional_losses_10433277inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЖBу
*__inference_skip_11_layer_call_fn_10433283inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЁBѓ
E__inference_skip_11_layer_call_and_return_conditional_losses_10433289inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ыBЬ
1__inference_concatenate_29_layer_call_fn_10433295inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
їBЅ
L__inference_concatenate_29_layer_call_and_return_conditional_losses_10433302inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
СBр
0__inference_expand_12_5x5_layer_call_fn_10433311inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 BЧ
K__inference_expand_12_5x5_layer_call_and_return_conditional_losses_10433322inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ТBс
2__inference_contract_12_3x3_layer_call_fn_10433331inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЂB■
M__inference_contract_12_3x3_layer_call_and_return_conditional_losses_10433342inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЖBу
*__inference_skip_12_layer_call_fn_10433348inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЁBѓ
E__inference_skip_12_layer_call_and_return_conditional_losses_10433354inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ыBЬ
1__inference_concatenate_30_layer_call_fn_10433360inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
їBЅ
L__inference_concatenate_30_layer_call_and_return_conditional_losses_10433367inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
СBр
0__inference_expand_13_5x5_layer_call_fn_10433376inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 BЧ
K__inference_expand_13_5x5_layer_call_and_return_conditional_losses_10433387inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ТBс
2__inference_contract_13_3x3_layer_call_fn_10433396inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЂB■
M__inference_contract_13_3x3_layer_call_and_return_conditional_losses_10433407inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЖBу
*__inference_skip_13_layer_call_fn_10433413inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЁBѓ
E__inference_skip_13_layer_call_and_return_conditional_losses_10433419inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ыBЬ
1__inference_concatenate_31_layer_call_fn_10433425inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
їBЅ
L__inference_concatenate_31_layer_call_and_return_conditional_losses_10433432inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
СBр
0__inference_expand_14_5x5_layer_call_fn_10433441inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 BЧ
K__inference_expand_14_5x5_layer_call_and_return_conditional_losses_10433452inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ТBс
2__inference_contract_14_3x3_layer_call_fn_10433461inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЂB■
M__inference_contract_14_3x3_layer_call_and_return_conditional_losses_10433472inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЖBу
*__inference_skip_14_layer_call_fn_10433478inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЁBѓ
E__inference_skip_14_layer_call_and_return_conditional_losses_10433484inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ыBЬ
1__inference_concatenate_32_layer_call_fn_10433490inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
їBЅ
L__inference_concatenate_32_layer_call_and_return_conditional_losses_10433497inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
СBр
0__inference_expand_15_5x5_layer_call_fn_10433506inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 BЧ
K__inference_expand_15_5x5_layer_call_and_return_conditional_losses_10433517inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ТBс
2__inference_contract_15_3x3_layer_call_fn_10433526inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЂB■
M__inference_contract_15_3x3_layer_call_and_return_conditional_losses_10433537inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЖBу
*__inference_skip_15_layer_call_fn_10433543inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЁBѓ
E__inference_skip_15_layer_call_and_return_conditional_losses_10433549inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ыBЬ
1__inference_concatenate_33_layer_call_fn_10433555inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
їBЅ
L__inference_concatenate_33_layer_call_and_return_conditional_losses_10433562inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
СBр
0__inference_expand_16_5x5_layer_call_fn_10433571inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 BЧ
K__inference_expand_16_5x5_layer_call_and_return_conditional_losses_10433582inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ТBс
2__inference_contract_16_3x3_layer_call_fn_10433591inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЂB■
M__inference_contract_16_3x3_layer_call_and_return_conditional_losses_10433602inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЖBу
*__inference_skip_16_layer_call_fn_10433608inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЁBѓ
E__inference_skip_16_layer_call_and_return_conditional_losses_10433614inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ыBЬ
1__inference_concatenate_34_layer_call_fn_10433620inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
їBЅ
L__inference_concatenate_34_layer_call_and_return_conditional_losses_10433627inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
СBр
0__inference_expand_17_5x5_layer_call_fn_10433636inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 BЧ
K__inference_expand_17_5x5_layer_call_and_return_conditional_losses_10433647inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ТBс
2__inference_contract_17_3x3_layer_call_fn_10433656inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЂB■
M__inference_contract_17_3x3_layer_call_and_return_conditional_losses_10433667inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЖBу
*__inference_skip_17_layer_call_fn_10433673inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЁBѓ
E__inference_skip_17_layer_call_and_return_conditional_losses_10433679inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ыBЬ
1__inference_concatenate_35_layer_call_fn_10433685inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
їBЅ
L__inference_concatenate_35_layer_call_and_return_conditional_losses_10433692inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
СBр
0__inference_expand_18_5x5_layer_call_fn_10433701inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 BЧ
K__inference_expand_18_5x5_layer_call_and_return_conditional_losses_10433712inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ТBс
2__inference_contract_18_3x3_layer_call_fn_10433721inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЂB■
M__inference_contract_18_3x3_layer_call_and_return_conditional_losses_10433732inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЖBу
*__inference_skip_18_layer_call_fn_10433738inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЁBѓ
E__inference_skip_18_layer_call_and_return_conditional_losses_10433744inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ыBЬ
1__inference_concatenate_36_layer_call_fn_10433750inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
їBЅ
L__inference_concatenate_36_layer_call_and_return_conditional_losses_10433757inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
СBр
0__inference_expand_19_5x5_layer_call_fn_10433766inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 BЧ
K__inference_expand_19_5x5_layer_call_and_return_conditional_losses_10433777inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ТBс
2__inference_contract_19_3x3_layer_call_fn_10433786inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЂB■
M__inference_contract_19_3x3_layer_call_and_return_conditional_losses_10433797inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЖBу
*__inference_skip_19_layer_call_fn_10433803inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЁBѓ
E__inference_skip_19_layer_call_and_return_conditional_losses_10433809inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ыBЬ
1__inference_concatenate_37_layer_call_fn_10433815inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
їBЅ
L__inference_concatenate_37_layer_call_and_return_conditional_losses_10433822inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
СBр
0__inference_expand_20_5x5_layer_call_fn_10433831inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 BЧ
K__inference_expand_20_5x5_layer_call_and_return_conditional_losses_10433842inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ТBс
2__inference_contract_20_3x3_layer_call_fn_10433851inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЂB■
M__inference_contract_20_3x3_layer_call_and_return_conditional_losses_10433862inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЖBу
*__inference_skip_20_layer_call_fn_10433868inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЁBѓ
E__inference_skip_20_layer_call_and_return_conditional_losses_10433874inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
УBт
4__inference_policy_aggregator_layer_call_fn_10433883inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЃBђ
O__inference_policy_aggregator_layer_call_and_return_conditional_losses_10433894inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЫB№
2__inference_all_value_input_layer_call_fn_10433900inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЇBі
M__inference_all_value_input_layer_call_and_return_conditional_losses_10433907inputs/0inputs/1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
0
О0
п1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
рBя
-__inference_border_off_layer_call_fn_10433916inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЧBщ
H__inference_border_off_layer_call_and_return_conditional_losses_10433926inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
уBС
3__inference_flat_value_input_layer_call_fn_10433931inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ѓB 
N__inference_flat_value_input_layer_call_and_return_conditional_losses_10433937inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
РB▀
.__inference_flat_logits_layer_call_fn_10433942inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
§BЩ
I__inference_flat_logits_layer_call_and_return_conditional_losses_10433948inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
№BВ
.__inference_policy_head_layer_call_fn_10433953inputs"»
д▓б
FullArgSpec%
argsџ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsб

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
іBЄ
I__inference_policy_head_layer_call_and_return_conditional_losses_10433958inputs"»
д▓б
FullArgSpec%
argsџ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsб

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
рBя
-__inference_value_head_layer_call_fn_10433967inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЧBщ
H__inference_value_head_layer_call_and_return_conditional_losses_10433978inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
R
р
	variables
Р
	keras_api

с
total

С
count"
_tf_keras_metric
0
с
0
С
1"
trackable_list_wrapper
.
р
	variables"
_generic_user_object
:  (2total
:  (2countѓ
#__inference__wrapped_model_10428931┌«stjk|}ЁєћЋЮъ▓│╗╝лЛ┘┌Ь№эЭїЇЋќфФ│┤╚╔ЛмТу№­ёЁЇјбБФг└┴╔╩я▀уУЧ§ЁєџЏБцИ╣┴┬оО▀ЯЗш§■њЊЏю░▒╣║╚╔ОпзЗ7б4
-б*
(і%
inputs         
ф "nфk
5
policy_head&і#
policy_head         ж
2

value_head$і!

value_head         ь
M__inference_all_value_input_layer_call_and_return_conditional_losses_10433907Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         	
ф "-б*
#і 
0         
џ ┼
2__inference_all_value_input_layer_call_fn_10433900јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         	
ф " і         ║
H__inference_border_off_layer_call_and_return_conditional_losses_10433926nОп7б4
-б*
(і%
inputs         
ф "-б*
#і 
0         
џ њ
-__inference_border_off_layer_call_fn_10433916aОп7б4
-б*
(і%
inputs         
ф " і         В
L__inference_concatenate_19_layer_call_and_return_conditional_losses_10432652Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф "-б*
#і 
0         	
џ ─
1__inference_concatenate_19_layer_call_fn_10432645јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф " і         	В
L__inference_concatenate_20_layer_call_and_return_conditional_losses_10432717Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф "-б*
#і 
0         	
џ ─
1__inference_concatenate_20_layer_call_fn_10432710јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф " і         	В
L__inference_concatenate_21_layer_call_and_return_conditional_losses_10432782Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф "-б*
#і 
0         	
џ ─
1__inference_concatenate_21_layer_call_fn_10432775јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф " і         	В
L__inference_concatenate_22_layer_call_and_return_conditional_losses_10432847Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф "-б*
#і 
0         	
џ ─
1__inference_concatenate_22_layer_call_fn_10432840јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф " і         	В
L__inference_concatenate_23_layer_call_and_return_conditional_losses_10432912Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф "-б*
#і 
0         	
џ ─
1__inference_concatenate_23_layer_call_fn_10432905јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф " і         	В
L__inference_concatenate_24_layer_call_and_return_conditional_losses_10432977Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф "-б*
#і 
0         	
џ ─
1__inference_concatenate_24_layer_call_fn_10432970јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф " і         	В
L__inference_concatenate_25_layer_call_and_return_conditional_losses_10433042Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф "-б*
#і 
0         	
џ ─
1__inference_concatenate_25_layer_call_fn_10433035јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф " і         	В
L__inference_concatenate_26_layer_call_and_return_conditional_losses_10433107Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф "-б*
#і 
0         	
џ ─
1__inference_concatenate_26_layer_call_fn_10433100јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф " і         	В
L__inference_concatenate_27_layer_call_and_return_conditional_losses_10433172Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф "-б*
#і 
0         	
џ ─
1__inference_concatenate_27_layer_call_fn_10433165јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф " і         	В
L__inference_concatenate_28_layer_call_and_return_conditional_losses_10433237Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф "-б*
#і 
0         	
џ ─
1__inference_concatenate_28_layer_call_fn_10433230јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф " і         	В
L__inference_concatenate_29_layer_call_and_return_conditional_losses_10433302Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф "-б*
#і 
0         	
џ ─
1__inference_concatenate_29_layer_call_fn_10433295јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф " і         	В
L__inference_concatenate_30_layer_call_and_return_conditional_losses_10433367Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф "-б*
#і 
0         	
џ ─
1__inference_concatenate_30_layer_call_fn_10433360јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф " і         	В
L__inference_concatenate_31_layer_call_and_return_conditional_losses_10433432Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф "-б*
#і 
0         	
џ ─
1__inference_concatenate_31_layer_call_fn_10433425јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф " і         	В
L__inference_concatenate_32_layer_call_and_return_conditional_losses_10433497Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф "-б*
#і 
0         	
џ ─
1__inference_concatenate_32_layer_call_fn_10433490јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф " і         	В
L__inference_concatenate_33_layer_call_and_return_conditional_losses_10433562Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф "-б*
#і 
0         	
џ ─
1__inference_concatenate_33_layer_call_fn_10433555јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф " і         	В
L__inference_concatenate_34_layer_call_and_return_conditional_losses_10433627Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф "-б*
#і 
0         	
џ ─
1__inference_concatenate_34_layer_call_fn_10433620јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф " і         	В
L__inference_concatenate_35_layer_call_and_return_conditional_losses_10433692Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф "-б*
#і 
0         	
џ ─
1__inference_concatenate_35_layer_call_fn_10433685јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф " і         	В
L__inference_concatenate_36_layer_call_and_return_conditional_losses_10433757Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф "-б*
#і 
0         	
џ ─
1__inference_concatenate_36_layer_call_fn_10433750јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф " і         	В
L__inference_concatenate_37_layer_call_and_return_conditional_losses_10433822Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф "-б*
#і 
0         	
џ ─
1__inference_concatenate_37_layer_call_fn_10433815јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф " і         	┐
M__inference_contract_10_3x3_layer_call_and_return_conditional_losses_10433212nЇј7б4
-б*
(і%
inputs          
ф "-б*
#і 
0         
џ Ќ
2__inference_contract_10_3x3_layer_call_fn_10433201aЇј7б4
-б*
(і%
inputs          
ф " і         ┐
M__inference_contract_11_3x3_layer_call_and_return_conditional_losses_10433277nФг7б4
-б*
(і%
inputs          
ф "-б*
#і 
0         
џ Ќ
2__inference_contract_11_3x3_layer_call_fn_10433266aФг7б4
-б*
(і%
inputs          
ф " і         ┐
M__inference_contract_12_3x3_layer_call_and_return_conditional_losses_10433342n╔╩7б4
-б*
(і%
inputs          
ф "-б*
#і 
0         
џ Ќ
2__inference_contract_12_3x3_layer_call_fn_10433331a╔╩7б4
-б*
(і%
inputs          
ф " і         ┐
M__inference_contract_13_3x3_layer_call_and_return_conditional_losses_10433407nуУ7б4
-б*
(і%
inputs          
ф "-б*
#і 
0         
џ Ќ
2__inference_contract_13_3x3_layer_call_fn_10433396aуУ7б4
-б*
(і%
inputs          
ф " і         ┐
M__inference_contract_14_3x3_layer_call_and_return_conditional_losses_10433472nЁє7б4
-б*
(і%
inputs          
ф "-б*
#і 
0         
џ Ќ
2__inference_contract_14_3x3_layer_call_fn_10433461aЁє7б4
-б*
(і%
inputs          
ф " і         ┐
M__inference_contract_15_3x3_layer_call_and_return_conditional_losses_10433537nБц7б4
-б*
(і%
inputs          
ф "-б*
#і 
0         
џ Ќ
2__inference_contract_15_3x3_layer_call_fn_10433526aБц7б4
-б*
(і%
inputs          
ф " і         ┐
M__inference_contract_16_3x3_layer_call_and_return_conditional_losses_10433602n┴┬7б4
-б*
(і%
inputs          
ф "-б*
#і 
0         
џ Ќ
2__inference_contract_16_3x3_layer_call_fn_10433591a┴┬7б4
-б*
(і%
inputs          
ф " і         ┐
M__inference_contract_17_3x3_layer_call_and_return_conditional_losses_10433667n▀Я7б4
-б*
(і%
inputs          
ф "-б*
#і 
0         
џ Ќ
2__inference_contract_17_3x3_layer_call_fn_10433656a▀Я7б4
-б*
(і%
inputs          
ф " і         ┐
M__inference_contract_18_3x3_layer_call_and_return_conditional_losses_10433732n§■7б4
-б*
(і%
inputs          
ф "-б*
#і 
0         
џ Ќ
2__inference_contract_18_3x3_layer_call_fn_10433721a§■7б4
-б*
(і%
inputs          
ф " і         ┐
M__inference_contract_19_3x3_layer_call_and_return_conditional_losses_10433797nЏю7б4
-б*
(і%
inputs          
ф "-б*
#і 
0         
џ Ќ
2__inference_contract_19_3x3_layer_call_fn_10433786aЏю7б4
-б*
(і%
inputs          
ф " і         ┐
L__inference_contract_1_5x5_layer_call_and_return_conditional_losses_10432639oЁє8б5
.б+
)і&
inputs         ђ
ф "-б*
#і 
0         
џ Ќ
1__inference_contract_1_5x5_layer_call_fn_10432628bЁє8б5
.б+
)і&
inputs         ђ
ф " і         ┐
M__inference_contract_20_3x3_layer_call_and_return_conditional_losses_10433862n╣║7б4
-б*
(і%
inputs          
ф "-б*
#і 
0         
џ Ќ
2__inference_contract_20_3x3_layer_call_fn_10433851a╣║7б4
-б*
(і%
inputs          
ф " і         Й
L__inference_contract_2_3x3_layer_call_and_return_conditional_losses_10432692nЮъ7б4
-б*
(і%
inputs          
ф "-б*
#і 
0         
џ ќ
1__inference_contract_2_3x3_layer_call_fn_10432681aЮъ7б4
-б*
(і%
inputs          
ф " і         Й
L__inference_contract_3_3x3_layer_call_and_return_conditional_losses_10432757n╗╝7б4
-б*
(і%
inputs          
ф "-б*
#і 
0         
џ ќ
1__inference_contract_3_3x3_layer_call_fn_10432746a╗╝7б4
-б*
(і%
inputs          
ф " і         Й
L__inference_contract_4_3x3_layer_call_and_return_conditional_losses_10432822n┘┌7б4
-б*
(і%
inputs          
ф "-б*
#і 
0         
џ ќ
1__inference_contract_4_3x3_layer_call_fn_10432811a┘┌7б4
-б*
(і%
inputs          
ф " і         Й
L__inference_contract_5_3x3_layer_call_and_return_conditional_losses_10432887nэЭ7б4
-б*
(і%
inputs          
ф "-б*
#і 
0         
џ ќ
1__inference_contract_5_3x3_layer_call_fn_10432876aэЭ7б4
-б*
(і%
inputs          
ф " і         Й
L__inference_contract_6_3x3_layer_call_and_return_conditional_losses_10432952nЋќ7б4
-б*
(і%
inputs          
ф "-б*
#і 
0         
џ ќ
1__inference_contract_6_3x3_layer_call_fn_10432941aЋќ7б4
-б*
(і%
inputs          
ф " і         Й
L__inference_contract_7_3x3_layer_call_and_return_conditional_losses_10433017n│┤7б4
-б*
(і%
inputs          
ф "-б*
#і 
0         
џ ќ
1__inference_contract_7_3x3_layer_call_fn_10433006a│┤7б4
-б*
(і%
inputs          
ф " і         Й
L__inference_contract_8_3x3_layer_call_and_return_conditional_losses_10433082nЛм7б4
-б*
(і%
inputs          
ф "-б*
#і 
0         
џ ќ
1__inference_contract_8_3x3_layer_call_fn_10433071aЛм7б4
-б*
(і%
inputs          
ф " і         Й
L__inference_contract_9_3x3_layer_call_and_return_conditional_losses_10433147n№­7б4
-б*
(і%
inputs          
ф "-б*
#і 
0         
џ ќ
1__inference_contract_9_3x3_layer_call_fn_10433136a№­7б4
-б*
(і%
inputs          
ф " і         й
K__inference_expand_10_5x5_layer_call_and_return_conditional_losses_10433192nёЁ7б4
-б*
(і%
inputs         	
ф "-б*
#і 
0          
џ Ћ
0__inference_expand_10_5x5_layer_call_fn_10433181aёЁ7б4
-б*
(і%
inputs         	
ф " і          й
K__inference_expand_11_5x5_layer_call_and_return_conditional_losses_10433257nбБ7б4
-б*
(і%
inputs         	
ф "-б*
#і 
0          
џ Ћ
0__inference_expand_11_5x5_layer_call_fn_10433246aбБ7б4
-б*
(і%
inputs         	
ф " і          й
K__inference_expand_12_5x5_layer_call_and_return_conditional_losses_10433322n└┴7б4
-б*
(і%
inputs         	
ф "-б*
#і 
0          
џ Ћ
0__inference_expand_12_5x5_layer_call_fn_10433311a└┴7б4
-б*
(і%
inputs         	
ф " і          й
K__inference_expand_13_5x5_layer_call_and_return_conditional_losses_10433387nя▀7б4
-б*
(і%
inputs         	
ф "-б*
#і 
0          
џ Ћ
0__inference_expand_13_5x5_layer_call_fn_10433376aя▀7б4
-б*
(і%
inputs         	
ф " і          й
K__inference_expand_14_5x5_layer_call_and_return_conditional_losses_10433452nЧ§7б4
-б*
(і%
inputs         	
ф "-б*
#і 
0          
џ Ћ
0__inference_expand_14_5x5_layer_call_fn_10433441aЧ§7б4
-б*
(і%
inputs         	
ф " і          й
K__inference_expand_15_5x5_layer_call_and_return_conditional_losses_10433517nџЏ7б4
-б*
(і%
inputs         	
ф "-б*
#і 
0          
џ Ћ
0__inference_expand_15_5x5_layer_call_fn_10433506aџЏ7б4
-б*
(і%
inputs         	
ф " і          й
K__inference_expand_16_5x5_layer_call_and_return_conditional_losses_10433582nИ╣7б4
-б*
(і%
inputs         	
ф "-б*
#і 
0          
џ Ћ
0__inference_expand_16_5x5_layer_call_fn_10433571aИ╣7б4
-б*
(і%
inputs         	
ф " і          й
K__inference_expand_17_5x5_layer_call_and_return_conditional_losses_10433647nоО7б4
-б*
(і%
inputs         	
ф "-б*
#і 
0          
џ Ћ
0__inference_expand_17_5x5_layer_call_fn_10433636aоО7б4
-б*
(і%
inputs         	
ф " і          й
K__inference_expand_18_5x5_layer_call_and_return_conditional_losses_10433712nЗш7б4
-б*
(і%
inputs         	
ф "-б*
#і 
0          
џ Ћ
0__inference_expand_18_5x5_layer_call_fn_10433701aЗш7б4
-б*
(і%
inputs         	
ф " і          й
K__inference_expand_19_5x5_layer_call_and_return_conditional_losses_10433777nњЊ7б4
-б*
(і%
inputs         	
ф "-б*
#і 
0          
џ Ћ
0__inference_expand_19_5x5_layer_call_fn_10433766aњЊ7б4
-б*
(і%
inputs         	
ф " і          й
L__inference_expand_1_11x11_layer_call_and_return_conditional_losses_10432599mst7б4
-б*
(і%
inputs         
ф ".б+
$і!
0         ђ
џ Ћ
1__inference_expand_1_11x11_layer_call_fn_10432588`st7б4
-б*
(і%
inputs         
ф "!і         ђй
K__inference_expand_20_5x5_layer_call_and_return_conditional_losses_10433842n░▒7б4
-б*
(і%
inputs         	
ф "-б*
#і 
0          
џ Ћ
0__inference_expand_20_5x5_layer_call_fn_10433831a░▒7б4
-б*
(і%
inputs         	
ф " і          ╝
J__inference_expand_2_5x5_layer_call_and_return_conditional_losses_10432672nћЋ7б4
-б*
(і%
inputs         	
ф "-б*
#і 
0          
џ ћ
/__inference_expand_2_5x5_layer_call_fn_10432661aћЋ7б4
-б*
(і%
inputs         	
ф " і          ╝
J__inference_expand_3_5x5_layer_call_and_return_conditional_losses_10432737n▓│7б4
-б*
(і%
inputs         	
ф "-б*
#і 
0          
џ ћ
/__inference_expand_3_5x5_layer_call_fn_10432726a▓│7б4
-б*
(і%
inputs         	
ф " і          ╝
J__inference_expand_4_5x5_layer_call_and_return_conditional_losses_10432802nлЛ7б4
-б*
(і%
inputs         	
ф "-б*
#і 
0          
џ ћ
/__inference_expand_4_5x5_layer_call_fn_10432791aлЛ7б4
-б*
(і%
inputs         	
ф " і          ╝
J__inference_expand_5_5x5_layer_call_and_return_conditional_losses_10432867nЬ№7б4
-б*
(і%
inputs         	
ф "-б*
#і 
0          
џ ћ
/__inference_expand_5_5x5_layer_call_fn_10432856aЬ№7б4
-б*
(і%
inputs         	
ф " і          ╝
J__inference_expand_6_5x5_layer_call_and_return_conditional_losses_10432932nїЇ7б4
-б*
(і%
inputs         	
ф "-б*
#і 
0          
џ ћ
/__inference_expand_6_5x5_layer_call_fn_10432921aїЇ7б4
-б*
(і%
inputs         	
ф " і          ╝
J__inference_expand_7_5x5_layer_call_and_return_conditional_losses_10432997nфФ7б4
-б*
(і%
inputs         	
ф "-б*
#і 
0          
џ ћ
/__inference_expand_7_5x5_layer_call_fn_10432986aфФ7б4
-б*
(і%
inputs         	
ф " і          ╝
J__inference_expand_8_5x5_layer_call_and_return_conditional_losses_10433062n╚╔7б4
-б*
(і%
inputs         	
ф "-б*
#і 
0          
џ ћ
/__inference_expand_8_5x5_layer_call_fn_10433051a╚╔7б4
-б*
(і%
inputs         	
ф " і          ╝
J__inference_expand_9_5x5_layer_call_and_return_conditional_losses_10433127nТу7б4
-б*
(і%
inputs         	
ф "-б*
#і 
0          
џ ћ
/__inference_expand_9_5x5_layer_call_fn_10433116aТу7б4
-б*
(і%
inputs         	
ф " і          «
I__inference_flat_logits_layer_call_and_return_conditional_losses_10433948a7б4
-б*
(і%
inputs         
ф "&б#
і
0         ж
џ є
.__inference_flat_logits_layer_call_fn_10433942T7б4
-б*
(і%
inputs         
ф "і         ж│
N__inference_flat_value_input_layer_call_and_return_conditional_losses_10433937a7б4
-б*
(і%
inputs         
ф "&б#
і
0         т,
џ І
3__inference_flat_value_input_layer_call_fn_10433931T7б4
-б*
(і%
inputs         
ф "і         т,њ
M__inference_gomoku_resnet_1_layer_call_and_return_conditional_losses_10432096└«stjk|}ЁєћЋЮъ▓│╗╝лЛ┘┌Ь№эЭїЇЋќфФ│┤╚╔ЛмТу№­ёЁЇјбБФг└┴╔╩я▀уУЧ§ЁєџЏБцИ╣┴┬оО▀ЯЗш§■њЊЏю░▒╣║╚╔ОпзЗ?б<
5б2
(і%
inputs         
p 

 
ф "LбI
Bџ?
і
0/0         ж
і
0/1         
џ њ
M__inference_gomoku_resnet_1_layer_call_and_return_conditional_losses_10432370└«stjk|}ЁєћЋЮъ▓│╗╝лЛ┘┌Ь№эЭїЇЋќфФ│┤╚╔ЛмТу№­ёЁЇјбБФг└┴╔╩я▀уУЧ§ЁєџЏБцИ╣┴┬оО▀ЯЗш§■њЊЏю░▒╣║╚╔ОпзЗ?б<
5б2
(і%
inputs         
p

 
ф "LбI
Bџ?
і
0/0         ж
і
0/1         
џ ж
2__inference_gomoku_resnet_1_layer_call_fn_10430246▓«stjk|}ЁєћЋЮъ▓│╗╝лЛ┘┌Ь№эЭїЇЋќфФ│┤╚╔ЛмТу№­ёЁЇјбБФг└┴╔╩я▀уУЧ§ЁєџЏБцИ╣┴┬оО▀ЯЗш§■њЊЏю░▒╣║╚╔ОпзЗ?б<
5б2
(і%
inputs         
p 

 
ф ">џ;
і
0         ж
і
1         ж
2__inference_gomoku_resnet_1_layer_call_fn_10431822▓«stjk|}ЁєћЋЮъ▓│╗╝лЛ┘┌Ь№эЭїЇЋќфФ│┤╚╔ЛмТу№­ёЁЇјбБФг└┴╔╩я▀уУЧ§ЁєџЏБцИ╣┴┬оО▀ЯЗш§■њЊЏю░▒╣║╚╔ОпзЗ?б<
5б2
(і%
inputs         
p

 
ф ">џ;
і
0         ж
і
1         ┴
P__inference_heuristic_detector_layer_call_and_return_conditional_losses_10432579mjk7б4
-б*
(і%
inputs         
ф ".б+
$і!
0         │
џ Ў
5__inference_heuristic_detector_layer_call_fn_10432568`jk7б4
-б*
(і%
inputs         
ф "!і         │┴
P__inference_heuristic_priority_layer_call_and_return_conditional_losses_10432619m|}8б5
.б+
)і&
inputs         │
ф "-б*
#і 
0         
џ Ў
5__inference_heuristic_priority_layer_call_fn_10432608`|}8б5
.б+
)і&
inputs         │
ф " і         ┴
O__inference_policy_aggregator_layer_call_and_return_conditional_losses_10433894n╚╔7б4
-б*
(і%
inputs         
ф "-б*
#і 
0         
џ Ў
4__inference_policy_aggregator_layer_call_fn_10433883a╚╔7б4
-б*
(і%
inputs         
ф " і         Ф
I__inference_policy_head_layer_call_and_return_conditional_losses_10433958^4б1
*б'
!і
inputs         ж

 
ф "&б#
і
0         ж
џ Ѓ
.__inference_policy_head_layer_call_fn_10433953Q4б1
*б'
!і
inputs         ж

 
ф "і         жЈ
&__inference_signature_wrapper_10432559С«stjk|}ЁєћЋЮъ▓│╗╝лЛ┘┌Ь№эЭїЇЋќфФ│┤╚╔ЛмТу№­ёЁЇјбБФг└┴╔╩я▀уУЧ§ЁєџЏБцИ╣┴┬оО▀ЯЗш§■њЊЏю░▒╣║╚╔ОпзЗAб>
б 
7ф4
2
inputs(і%
inputs         "nфk
5
policy_head&і#
policy_head         ж
2

value_head$і!

value_head         т
E__inference_skip_10_layer_call_and_return_conditional_losses_10433224Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф "-б*
#і 
0         
џ й
*__inference_skip_10_layer_call_fn_10433218јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф " і         т
E__inference_skip_11_layer_call_and_return_conditional_losses_10433289Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф "-б*
#і 
0         
џ й
*__inference_skip_11_layer_call_fn_10433283јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф " і         т
E__inference_skip_12_layer_call_and_return_conditional_losses_10433354Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф "-б*
#і 
0         
џ й
*__inference_skip_12_layer_call_fn_10433348јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф " і         т
E__inference_skip_13_layer_call_and_return_conditional_losses_10433419Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф "-б*
#і 
0         
џ й
*__inference_skip_13_layer_call_fn_10433413јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф " і         т
E__inference_skip_14_layer_call_and_return_conditional_losses_10433484Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф "-б*
#і 
0         
џ й
*__inference_skip_14_layer_call_fn_10433478јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф " і         т
E__inference_skip_15_layer_call_and_return_conditional_losses_10433549Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф "-б*
#і 
0         
џ й
*__inference_skip_15_layer_call_fn_10433543јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф " і         т
E__inference_skip_16_layer_call_and_return_conditional_losses_10433614Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф "-б*
#і 
0         
џ й
*__inference_skip_16_layer_call_fn_10433608јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф " і         т
E__inference_skip_17_layer_call_and_return_conditional_losses_10433679Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф "-б*
#і 
0         
џ й
*__inference_skip_17_layer_call_fn_10433673јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф " і         т
E__inference_skip_18_layer_call_and_return_conditional_losses_10433744Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф "-б*
#і 
0         
џ й
*__inference_skip_18_layer_call_fn_10433738јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф " і         т
E__inference_skip_19_layer_call_and_return_conditional_losses_10433809Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф "-б*
#і 
0         
џ й
*__inference_skip_19_layer_call_fn_10433803јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф " і         т
E__inference_skip_20_layer_call_and_return_conditional_losses_10433874Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф "-б*
#і 
0         
џ й
*__inference_skip_20_layer_call_fn_10433868јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф " і         С
D__inference_skip_2_layer_call_and_return_conditional_losses_10432704Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф "-б*
#і 
0         
џ ╝
)__inference_skip_2_layer_call_fn_10432698јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф " і         С
D__inference_skip_3_layer_call_and_return_conditional_losses_10432769Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф "-б*
#і 
0         
џ ╝
)__inference_skip_3_layer_call_fn_10432763јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф " і         С
D__inference_skip_4_layer_call_and_return_conditional_losses_10432834Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф "-б*
#і 
0         
џ ╝
)__inference_skip_4_layer_call_fn_10432828јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф " і         С
D__inference_skip_5_layer_call_and_return_conditional_losses_10432899Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф "-б*
#і 
0         
џ ╝
)__inference_skip_5_layer_call_fn_10432893јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф " і         С
D__inference_skip_6_layer_call_and_return_conditional_losses_10432964Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф "-б*
#і 
0         
џ ╝
)__inference_skip_6_layer_call_fn_10432958јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф " і         С
D__inference_skip_7_layer_call_and_return_conditional_losses_10433029Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф "-б*
#і 
0         
џ ╝
)__inference_skip_7_layer_call_fn_10433023јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф " і         С
D__inference_skip_8_layer_call_and_return_conditional_losses_10433094Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф "-б*
#і 
0         
џ ╝
)__inference_skip_8_layer_call_fn_10433088јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф " і         С
D__inference_skip_9_layer_call_and_return_conditional_losses_10433159Џjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф "-б*
#і 
0         
џ ╝
)__inference_skip_9_layer_call_fn_10433153јjбg
`б]
[џX
*і'
inputs/0         
*і'
inputs/1         
ф " і         Ф
H__inference_value_head_layer_call_and_return_conditional_losses_10433978_зЗ0б-
&б#
!і
inputs         т,
ф "%б"
і
0         
џ Ѓ
-__inference_value_head_layer_call_fn_10433967RзЗ0б-
&б#
!і
inputs         т,
ф "і         
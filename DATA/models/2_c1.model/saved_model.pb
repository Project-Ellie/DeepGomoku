��9
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
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
�
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
delete_old_dirsbool(�
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
dtypetype�
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
�
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
executor_typestring ��
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
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.9.12unknown8ټ,
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
shape:	�,*"
shared_namevalue_head/kernel
x
%value_head/kernel/Read/ReadVariableOpReadVariableOpvalue_head/kernel*
_output_shapes
:	�,*
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
�
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
�
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
�
policy_aggregator/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namepolicy_aggregator/kernel
�
,policy_aggregator/kernel/Read/ReadVariableOpReadVariableOppolicy_aggregator/kernel*&
_output_shapes
:*
dtype0
�
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
�
contract_20_3x3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namecontract_20_3x3/kernel
�
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
�
expand_20_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *%
shared_nameexpand_20_5x5/kernel
�
(expand_20_5x5/kernel/Read/ReadVariableOpReadVariableOpexpand_20_5x5/kernel*&
_output_shapes
:	 *
dtype0
�
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
�
contract_19_3x3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namecontract_19_3x3/kernel
�
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
�
expand_19_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *%
shared_nameexpand_19_5x5/kernel
�
(expand_19_5x5/kernel/Read/ReadVariableOpReadVariableOpexpand_19_5x5/kernel*&
_output_shapes
:	 *
dtype0
�
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
�
contract_18_3x3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namecontract_18_3x3/kernel
�
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
�
expand_18_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *%
shared_nameexpand_18_5x5/kernel
�
(expand_18_5x5/kernel/Read/ReadVariableOpReadVariableOpexpand_18_5x5/kernel*&
_output_shapes
:	 *
dtype0
�
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
�
contract_17_3x3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namecontract_17_3x3/kernel
�
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
�
expand_17_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *%
shared_nameexpand_17_5x5/kernel
�
(expand_17_5x5/kernel/Read/ReadVariableOpReadVariableOpexpand_17_5x5/kernel*&
_output_shapes
:	 *
dtype0
�
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
�
contract_16_3x3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namecontract_16_3x3/kernel
�
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
�
expand_16_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *%
shared_nameexpand_16_5x5/kernel
�
(expand_16_5x5/kernel/Read/ReadVariableOpReadVariableOpexpand_16_5x5/kernel*&
_output_shapes
:	 *
dtype0
�
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
�
contract_15_3x3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namecontract_15_3x3/kernel
�
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
�
expand_15_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *%
shared_nameexpand_15_5x5/kernel
�
(expand_15_5x5/kernel/Read/ReadVariableOpReadVariableOpexpand_15_5x5/kernel*&
_output_shapes
:	 *
dtype0
�
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
�
contract_14_3x3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namecontract_14_3x3/kernel
�
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
�
expand_14_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *%
shared_nameexpand_14_5x5/kernel
�
(expand_14_5x5/kernel/Read/ReadVariableOpReadVariableOpexpand_14_5x5/kernel*&
_output_shapes
:	 *
dtype0
�
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
�
contract_13_3x3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namecontract_13_3x3/kernel
�
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
�
expand_13_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *%
shared_nameexpand_13_5x5/kernel
�
(expand_13_5x5/kernel/Read/ReadVariableOpReadVariableOpexpand_13_5x5/kernel*&
_output_shapes
:	 *
dtype0
�
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
�
contract_12_3x3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namecontract_12_3x3/kernel
�
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
�
expand_12_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *%
shared_nameexpand_12_5x5/kernel
�
(expand_12_5x5/kernel/Read/ReadVariableOpReadVariableOpexpand_12_5x5/kernel*&
_output_shapes
:	 *
dtype0
�
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
�
contract_11_3x3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namecontract_11_3x3/kernel
�
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
�
expand_11_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *%
shared_nameexpand_11_5x5/kernel
�
(expand_11_5x5/kernel/Read/ReadVariableOpReadVariableOpexpand_11_5x5/kernel*&
_output_shapes
:	 *
dtype0
�
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
�
contract_10_3x3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namecontract_10_3x3/kernel
�
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
�
expand_10_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *%
shared_nameexpand_10_5x5/kernel
�
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
�
contract_9_3x3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_namecontract_9_3x3/kernel
�
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
�
expand_9_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *$
shared_nameexpand_9_5x5/kernel
�
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
�
contract_8_3x3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_namecontract_8_3x3/kernel
�
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
�
expand_8_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *$
shared_nameexpand_8_5x5/kernel
�
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
�
contract_7_3x3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_namecontract_7_3x3/kernel
�
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
�
expand_7_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *$
shared_nameexpand_7_5x5/kernel
�
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
�
contract_6_3x3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_namecontract_6_3x3/kernel
�
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
�
expand_6_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *$
shared_nameexpand_6_5x5/kernel
�
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
�
contract_5_3x3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_namecontract_5_3x3/kernel
�
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
�
expand_5_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *$
shared_nameexpand_5_5x5/kernel
�
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
�
contract_4_3x3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_namecontract_4_3x3/kernel
�
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
�
expand_4_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *$
shared_nameexpand_4_5x5/kernel
�
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
�
contract_3_3x3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_namecontract_3_3x3/kernel
�
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
�
expand_3_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *$
shared_nameexpand_3_5x5/kernel
�
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
�
contract_2_3x3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_namecontract_2_3x3/kernel
�
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
�
expand_2_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *$
shared_nameexpand_2_5x5/kernel
�
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
�
contract_1_5x5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_namecontract_1_5x5/kernel
�
)contract_1_5x5/kernel/Read/ReadVariableOpReadVariableOpcontract_1_5x5/kernel*'
_output_shapes
:�*
dtype0
�
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
�
heuristic_priority/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_nameheuristic_priority/kernel
�
-heuristic_priority/kernel/Read/ReadVariableOpReadVariableOpheuristic_priority/kernel*'
_output_shapes
:�*
dtype0

expand_1_11x11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameexpand_1_11x11/bias
x
'expand_1_11x11/bias/Read/ReadVariableOpReadVariableOpexpand_1_11x11/bias*
_output_shapes	
:�*
dtype0
�
expand_1_11x11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameexpand_1_11x11/kernel
�
)expand_1_11x11/kernel/Read/ReadVariableOpReadVariableOpexpand_1_11x11/kernel*'
_output_shapes
:�*
dtype0
�
heuristic_detector/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*(
shared_nameheuristic_detector/bias
�
+heuristic_detector/bias/Read/ReadVariableOpReadVariableOpheuristic_detector/bias*
_output_shapes	
:�*
dtype0
�
heuristic_detector/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_nameheuristic_detector/kernel
�
-heuristic_detector/kernel/Read/ReadVariableOpReadVariableOpheuristic_detector/kernel*'
_output_shapes
:�*
dtype0

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
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
�
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses

jkernel
kbias
 l_jit_compiled_convolution_op*
�
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses

skernel
tbias
 u_jit_compiled_convolution_op*
�
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses

|kernel
}bias
 ~_jit_compiled_convolution_op*
�
	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 

�	keras_api* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
j0
k1
s2
t3
|4
}5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
�58
�59
�60
�61
�62
�63
�64
�65
�66
�67
�68
�69
�70
�71
�72
�73
�74
�75
�76
�77
�78
�79
�80
�81
�82
�83
�84
�85
�86
�87
�88
�89*
�
s0
t1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
�58
�59
�60
�61
�62
�63
�64
�65
�66
�67
�68
�69
�70
�71
�72
�73
�74
�75
�76
�77
�78
�79
�80
�81
�82
�83*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
`_default_save_signature
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 

�serving_default* 

j0
k1*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
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
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
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
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
ic
VARIABLE_VALUEheuristic_priority/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEheuristic_priority/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
e_
VARIABLE_VALUEcontract_1_5x5/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEcontract_1_5x5/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
c]
VARIABLE_VALUEexpand_2_5x5/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEexpand_2_5x5/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
e_
VARIABLE_VALUEcontract_2_3x3/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEcontract_2_3x3/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
c]
VARIABLE_VALUEexpand_3_5x5/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEexpand_3_5x5/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
e_
VARIABLE_VALUEcontract_3_3x3/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEcontract_3_3x3/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
c]
VARIABLE_VALUEexpand_4_5x5/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEexpand_4_5x5/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
e_
VARIABLE_VALUEcontract_4_3x3/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEcontract_4_3x3/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
d^
VARIABLE_VALUEexpand_5_5x5/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEexpand_5_5x5/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
f`
VARIABLE_VALUEcontract_5_3x3/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEcontract_5_3x3/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
d^
VARIABLE_VALUEexpand_6_5x5/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEexpand_6_5x5/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
f`
VARIABLE_VALUEcontract_6_3x3/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEcontract_6_3x3/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
d^
VARIABLE_VALUEexpand_7_5x5/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEexpand_7_5x5/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
f`
VARIABLE_VALUEcontract_7_3x3/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEcontract_7_3x3/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
d^
VARIABLE_VALUEexpand_8_5x5/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEexpand_8_5x5/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
f`
VARIABLE_VALUEcontract_8_3x3/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEcontract_8_3x3/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
d^
VARIABLE_VALUEexpand_9_5x5/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEexpand_9_5x5/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
f`
VARIABLE_VALUEcontract_9_3x3/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEcontract_9_3x3/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
e_
VARIABLE_VALUEexpand_10_5x5/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEexpand_10_5x5/bias5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
ga
VARIABLE_VALUEcontract_10_3x3/kernel7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEcontract_10_3x3/bias5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
e_
VARIABLE_VALUEexpand_11_5x5/kernel7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEexpand_11_5x5/bias5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
ga
VARIABLE_VALUEcontract_11_3x3/kernel7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEcontract_11_3x3/bias5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
e_
VARIABLE_VALUEexpand_12_5x5/kernel7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEexpand_12_5x5/bias5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
ga
VARIABLE_VALUEcontract_12_3x3/kernel7layer_with_weights-25/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEcontract_12_3x3/bias5layer_with_weights-25/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
e_
VARIABLE_VALUEexpand_13_5x5/kernel7layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEexpand_13_5x5/bias5layer_with_weights-26/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
ga
VARIABLE_VALUEcontract_13_3x3/kernel7layer_with_weights-27/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEcontract_13_3x3/bias5layer_with_weights-27/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
e_
VARIABLE_VALUEexpand_14_5x5/kernel7layer_with_weights-28/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEexpand_14_5x5/bias5layer_with_weights-28/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
ga
VARIABLE_VALUEcontract_14_3x3/kernel7layer_with_weights-29/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEcontract_14_3x3/bias5layer_with_weights-29/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�	non_trainable_variables
�	layers
�	metrics
 �	layer_regularization_losses
�	layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�	trace_0* 

�	trace_0* 
* 
* 
* 
�
�	non_trainable_variables
�	layers
�	metrics
 �	layer_regularization_losses
�	layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�	trace_0* 

�	trace_0* 

�0
�1*

�0
�1*
* 
�
�	non_trainable_variables
�	layers
�	metrics
 �	layer_regularization_losses
�	layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�	trace_0* 

�	trace_0* 
e_
VARIABLE_VALUEexpand_15_5x5/kernel7layer_with_weights-30/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEexpand_15_5x5/bias5layer_with_weights-30/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�	non_trainable_variables
�	layers
�	metrics
 �	layer_regularization_losses
�	layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�	trace_0* 

�	trace_0* 
ga
VARIABLE_VALUEcontract_15_3x3/kernel7layer_with_weights-31/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEcontract_15_3x3/bias5layer_with_weights-31/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�	non_trainable_variables
�	layers
�	metrics
 �	layer_regularization_losses
�	layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�	trace_0* 

�	trace_0* 
* 
* 
* 
�
�	non_trainable_variables
�	layers
�	metrics
 �	layer_regularization_losses
�	layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�	trace_0* 

�	trace_0* 

�0
�1*

�0
�1*
* 
�
�	non_trainable_variables
�	layers
�	metrics
 �	layer_regularization_losses
�	layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�	trace_0* 

�	trace_0* 
e_
VARIABLE_VALUEexpand_16_5x5/kernel7layer_with_weights-32/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEexpand_16_5x5/bias5layer_with_weights-32/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�	non_trainable_variables
�	layers
�	metrics
 �	layer_regularization_losses
�	layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�	trace_0* 

�	trace_0* 
ga
VARIABLE_VALUEcontract_16_3x3/kernel7layer_with_weights-33/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEcontract_16_3x3/bias5layer_with_weights-33/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�	non_trainable_variables
�	layers
�	metrics
 �	layer_regularization_losses
�	layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�	trace_0* 

�	trace_0* 
* 
* 
* 
�
�	non_trainable_variables
�	layers
�	metrics
 �	layer_regularization_losses
�	layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�	trace_0* 

�	trace_0* 

�0
�1*

�0
�1*
* 
�
�	non_trainable_variables
�	layers
�	metrics
 �	layer_regularization_losses
�	layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�	trace_0* 

�	trace_0* 
e_
VARIABLE_VALUEexpand_17_5x5/kernel7layer_with_weights-34/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEexpand_17_5x5/bias5layer_with_weights-34/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�	non_trainable_variables
�	layers
�	metrics
 �	layer_regularization_losses
�	layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�	trace_0* 

�	trace_0* 
ga
VARIABLE_VALUEcontract_17_3x3/kernel7layer_with_weights-35/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEcontract_17_3x3/bias5layer_with_weights-35/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�	non_trainable_variables
�	layers
�	metrics
 �	layer_regularization_losses
�	layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�	trace_0* 

�	trace_0* 
* 
* 
* 
�
�	non_trainable_variables
�	layers
�	metrics
 �	layer_regularization_losses
�	layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�	trace_0* 

�	trace_0* 

�0
�1*

�0
�1*
* 
�
�	non_trainable_variables
�	layers
�	metrics
 �	layer_regularization_losses
�	layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�	trace_0* 

�	trace_0* 
e_
VARIABLE_VALUEexpand_18_5x5/kernel7layer_with_weights-36/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEexpand_18_5x5/bias5layer_with_weights-36/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�	non_trainable_variables
�	layers
�	metrics
 �	layer_regularization_losses
�	layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�	trace_0* 

�	trace_0* 
ga
VARIABLE_VALUEcontract_18_3x3/kernel7layer_with_weights-37/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEcontract_18_3x3/bias5layer_with_weights-37/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�	non_trainable_variables
�	layers
�	metrics
 �	layer_regularization_losses
�	layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�	trace_0* 

�	trace_0* 
* 
* 
* 
�
�	non_trainable_variables
�	layers
�	metrics
 �	layer_regularization_losses
�	layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�	trace_0* 

�	trace_0* 

�0
�1*

�0
�1*
* 
�
�	non_trainable_variables
�	layers
�
metrics
 �
layer_regularization_losses
�
layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�
trace_0* 

�
trace_0* 
e_
VARIABLE_VALUEexpand_19_5x5/kernel7layer_with_weights-38/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEexpand_19_5x5/bias5layer_with_weights-38/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�
non_trainable_variables
�
layers
�
metrics
 �
layer_regularization_losses
�
layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�
trace_0* 

�
trace_0* 
ga
VARIABLE_VALUEcontract_19_3x3/kernel7layer_with_weights-39/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEcontract_19_3x3/bias5layer_with_weights-39/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�
non_trainable_variables
�
layers
�
metrics
 �
layer_regularization_losses
�
layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�
trace_0* 

�
trace_0* 
* 
* 
* 
�
�
non_trainable_variables
�
layers
�
metrics
 �
layer_regularization_losses
�
layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�
trace_0* 

�
trace_0* 

�0
�1*

�0
�1*
* 
�
�
non_trainable_variables
�
layers
�
metrics
 �
layer_regularization_losses
�
layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�
trace_0* 

�
trace_0* 
e_
VARIABLE_VALUEexpand_20_5x5/kernel7layer_with_weights-40/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEexpand_20_5x5/bias5layer_with_weights-40/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�
non_trainable_variables
�
layers
�
metrics
 �
layer_regularization_losses
�
layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�
trace_0* 

�
trace_0* 
ga
VARIABLE_VALUEcontract_20_3x3/kernel7layer_with_weights-41/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEcontract_20_3x3/bias5layer_with_weights-41/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�
non_trainable_variables
�
layers
�
metrics
 �
layer_regularization_losses
�
layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�
trace_0* 

�
trace_0* 

�0
�1*

�0
�1*
* 
�
�
non_trainable_variables
�
layers
�
metrics
 �
layer_regularization_losses
�
layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�
trace_0* 

�
trace_0* 
ic
VARIABLE_VALUEpolicy_aggregator/kernel7layer_with_weights-42/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEpolicy_aggregator/bias5layer_with_weights-42/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�
non_trainable_variables
�
layers
�
metrics
 �
layer_regularization_losses
�
layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�
trace_0* 

�
trace_0* 

�0
�1*
* 
* 
�
�
non_trainable_variables
�
layers
�
metrics
 �
layer_regularization_losses
�
layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�
trace_0* 

�
trace_0* 
b\
VARIABLE_VALUEborder_off/kernel7layer_with_weights-43/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEborder_off/bias5layer_with_weights-43/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�
non_trainable_variables
�
layers
�
metrics
 �
layer_regularization_losses
�
layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�
trace_0* 

�
trace_0* 
* 
* 
* 
�
�
non_trainable_variables
�
layers
�
metrics
 �
layer_regularization_losses
�
layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�
trace_0* 

�
trace_0* 
* 
* 
* 
* 
�
�
non_trainable_variables
�
layers
�
metrics
 �
layer_regularization_losses
�
layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�
trace_0* 

�
trace_0* 

�0
�1*

�0
�1*
* 
�
�
non_trainable_variables
�
layers
�
metrics
 �
layer_regularization_losses
�
layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�
trace_0* 

�
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
�4
�5*
�
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

�
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
�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
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
�
	variables
�
	keras_api

�
total

�
count*

�
0
�
1*

�
	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
�
serving_default_inputsPlaceholder*/
_output_shapes
:���������*
dtype0*$
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputsexpand_1_11x11/kernelexpand_1_11x11/biasheuristic_detector/kernelheuristic_detector/biasheuristic_priority/kernelheuristic_priority/biascontract_1_5x5/kernelcontract_1_5x5/biasexpand_2_5x5/kernelexpand_2_5x5/biascontract_2_3x3/kernelcontract_2_3x3/biasexpand_3_5x5/kernelexpand_3_5x5/biascontract_3_3x3/kernelcontract_3_3x3/biasexpand_4_5x5/kernelexpand_4_5x5/biascontract_4_3x3/kernelcontract_4_3x3/biasexpand_5_5x5/kernelexpand_5_5x5/biascontract_5_3x3/kernelcontract_5_3x3/biasexpand_6_5x5/kernelexpand_6_5x5/biascontract_6_3x3/kernelcontract_6_3x3/biasexpand_7_5x5/kernelexpand_7_5x5/biascontract_7_3x3/kernelcontract_7_3x3/biasexpand_8_5x5/kernelexpand_8_5x5/biascontract_8_3x3/kernelcontract_8_3x3/biasexpand_9_5x5/kernelexpand_9_5x5/biascontract_9_3x3/kernelcontract_9_3x3/biasexpand_10_5x5/kernelexpand_10_5x5/biascontract_10_3x3/kernelcontract_10_3x3/biasexpand_11_5x5/kernelexpand_11_5x5/biascontract_11_3x3/kernelcontract_11_3x3/biasexpand_12_5x5/kernelexpand_12_5x5/biascontract_12_3x3/kernelcontract_12_3x3/biasexpand_13_5x5/kernelexpand_13_5x5/biascontract_13_3x3/kernelcontract_13_3x3/biasexpand_14_5x5/kernelexpand_14_5x5/biascontract_14_3x3/kernelcontract_14_3x3/biasexpand_15_5x5/kernelexpand_15_5x5/biascontract_15_3x3/kernelcontract_15_3x3/biasexpand_16_5x5/kernelexpand_16_5x5/biascontract_16_3x3/kernelcontract_16_3x3/biasexpand_17_5x5/kernelexpand_17_5x5/biascontract_17_3x3/kernelcontract_17_3x3/biasexpand_18_5x5/kernelexpand_18_5x5/biascontract_18_3x3/kernelcontract_18_3x3/biasexpand_19_5x5/kernelexpand_19_5x5/biascontract_19_3x3/kernelcontract_19_3x3/biasexpand_20_5x5/kernelexpand_20_5x5/biascontract_20_3x3/kernelcontract_20_3x3/biaspolicy_aggregator/kernelpolicy_aggregator/biasborder_off/kernelborder_off/biasvalue_head/kernelvalue_head/bias*f
Tin_
]2[*
Tout
2*
_collective_manager_ids
 *;
_output_shapes)
':����������:���������*|
_read_only_resource_inputs^
\Z	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ*0
config_proto 

CPU

GPU2*0J 8� *.
f)R'
%__inference_signature_wrapper_3617857
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
� 
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
GPU2*0J 8� *)
f$R"
 __inference__traced_save_3619576
�
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
GPU2*0J 8� *,
f'R%
#__inference__traced_restore_3619862��&
�
�
0__inference_contract_6_3x3_layer_call_fn_3618239

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_6_3x3_layer_call_and_return_conditional_losses_3614545w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
n
D__inference_skip_10_layer_call_and_return_conditional_losses_3614761

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
w
K__inference_concatenate_14_layer_call_and_return_conditional_losses_3618860
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
:���������	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
u
K__inference_concatenate_12_layer_call_and_return_conditional_losses_3614923

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
:���������	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
w
K__inference_concatenate_16_layer_call_and_return_conditional_losses_3618990
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
:���������	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
u
K__inference_concatenate_15_layer_call_and_return_conditional_losses_3615076

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
:���������	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
O__inference_heuristic_detector_layer_call_and_return_conditional_losses_3614264

inputs9
conv2d_readvariableop_resource:�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
d
H__inference_policy_head_layer_call_and_return_conditional_losses_3615355

inputs
identityM
SoftmaxSoftmaxinputs*
T0*(
_output_shapes
:����������Z
IdentityIdentitySoftmax:softmax:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
Ё
�-
J__inference_gomoku_resnet_layer_call_and_return_conditional_losses_3616748

inputs1
expand_1_11x11_3616477:�%
expand_1_11x11_3616479:	�5
heuristic_detector_3616482:�)
heuristic_detector_3616484:	�5
heuristic_priority_3616487:�(
heuristic_priority_3616489:1
contract_1_5x5_3616492:�$
contract_1_5x5_3616494:.
expand_2_5x5_3616498:	 "
expand_2_5x5_3616500: 0
contract_2_3x3_3616503: $
contract_2_3x3_3616505:.
expand_3_5x5_3616510:	 "
expand_3_5x5_3616512: 0
contract_3_3x3_3616515: $
contract_3_3x3_3616517:.
expand_4_5x5_3616522:	 "
expand_4_5x5_3616524: 0
contract_4_3x3_3616527: $
contract_4_3x3_3616529:.
expand_5_5x5_3616534:	 "
expand_5_5x5_3616536: 0
contract_5_3x3_3616539: $
contract_5_3x3_3616541:.
expand_6_5x5_3616546:	 "
expand_6_5x5_3616548: 0
contract_6_3x3_3616551: $
contract_6_3x3_3616553:.
expand_7_5x5_3616558:	 "
expand_7_5x5_3616560: 0
contract_7_3x3_3616563: $
contract_7_3x3_3616565:.
expand_8_5x5_3616570:	 "
expand_8_5x5_3616572: 0
contract_8_3x3_3616575: $
contract_8_3x3_3616577:.
expand_9_5x5_3616582:	 "
expand_9_5x5_3616584: 0
contract_9_3x3_3616587: $
contract_9_3x3_3616589:/
expand_10_5x5_3616594:	 #
expand_10_5x5_3616596: 1
contract_10_3x3_3616599: %
contract_10_3x3_3616601:/
expand_11_5x5_3616606:	 #
expand_11_5x5_3616608: 1
contract_11_3x3_3616611: %
contract_11_3x3_3616613:/
expand_12_5x5_3616618:	 #
expand_12_5x5_3616620: 1
contract_12_3x3_3616623: %
contract_12_3x3_3616625:/
expand_13_5x5_3616630:	 #
expand_13_5x5_3616632: 1
contract_13_3x3_3616635: %
contract_13_3x3_3616637:/
expand_14_5x5_3616642:	 #
expand_14_5x5_3616644: 1
contract_14_3x3_3616647: %
contract_14_3x3_3616649:/
expand_15_5x5_3616654:	 #
expand_15_5x5_3616656: 1
contract_15_3x3_3616659: %
contract_15_3x3_3616661:/
expand_16_5x5_3616666:	 #
expand_16_5x5_3616668: 1
contract_16_3x3_3616671: %
contract_16_3x3_3616673:/
expand_17_5x5_3616678:	 #
expand_17_5x5_3616680: 1
contract_17_3x3_3616683: %
contract_17_3x3_3616685:/
expand_18_5x5_3616690:	 #
expand_18_5x5_3616692: 1
contract_18_3x3_3616695: %
contract_18_3x3_3616697:/
expand_19_5x5_3616702:	 #
expand_19_5x5_3616704: 1
contract_19_3x3_3616707: %
contract_19_3x3_3616709:/
expand_20_5x5_3616714:	 #
expand_20_5x5_3616716: 1
contract_20_3x3_3616719: %
contract_20_3x3_3616721:3
policy_aggregator_3616726:'
policy_aggregator_3616728:,
border_off_3616732: 
border_off_3616734:%
value_head_3616740:	�, 
value_head_3616742:
identity

identity_1��"border_off/StatefulPartitionedCall�'contract_10_3x3/StatefulPartitionedCall�'contract_11_3x3/StatefulPartitionedCall�'contract_12_3x3/StatefulPartitionedCall�'contract_13_3x3/StatefulPartitionedCall�'contract_14_3x3/StatefulPartitionedCall�'contract_15_3x3/StatefulPartitionedCall�'contract_16_3x3/StatefulPartitionedCall�'contract_17_3x3/StatefulPartitionedCall�'contract_18_3x3/StatefulPartitionedCall�'contract_19_3x3/StatefulPartitionedCall�&contract_1_5x5/StatefulPartitionedCall�'contract_20_3x3/StatefulPartitionedCall�&contract_2_3x3/StatefulPartitionedCall�&contract_3_3x3/StatefulPartitionedCall�&contract_4_3x3/StatefulPartitionedCall�&contract_5_3x3/StatefulPartitionedCall�&contract_6_3x3/StatefulPartitionedCall�&contract_7_3x3/StatefulPartitionedCall�&contract_8_3x3/StatefulPartitionedCall�&contract_9_3x3/StatefulPartitionedCall�%expand_10_5x5/StatefulPartitionedCall�%expand_11_5x5/StatefulPartitionedCall�%expand_12_5x5/StatefulPartitionedCall�%expand_13_5x5/StatefulPartitionedCall�%expand_14_5x5/StatefulPartitionedCall�%expand_15_5x5/StatefulPartitionedCall�%expand_16_5x5/StatefulPartitionedCall�%expand_17_5x5/StatefulPartitionedCall�%expand_18_5x5/StatefulPartitionedCall�%expand_19_5x5/StatefulPartitionedCall�&expand_1_11x11/StatefulPartitionedCall�%expand_20_5x5/StatefulPartitionedCall�$expand_2_5x5/StatefulPartitionedCall�$expand_3_5x5/StatefulPartitionedCall�$expand_4_5x5/StatefulPartitionedCall�$expand_5_5x5/StatefulPartitionedCall�$expand_6_5x5/StatefulPartitionedCall�$expand_7_5x5/StatefulPartitionedCall�$expand_8_5x5/StatefulPartitionedCall�$expand_9_5x5/StatefulPartitionedCall�*heuristic_detector/StatefulPartitionedCall�*heuristic_priority/StatefulPartitionedCall�)policy_aggregator/StatefulPartitionedCall�"value_head/StatefulPartitionedCall�
&expand_1_11x11/StatefulPartitionedCallStatefulPartitionedCallinputsexpand_1_11x11_3616477expand_1_11x11_3616479*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_expand_1_11x11_layer_call_and_return_conditional_losses_3614247�
*heuristic_detector/StatefulPartitionedCallStatefulPartitionedCallinputsheuristic_detector_3616482heuristic_detector_3616484*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_heuristic_detector_layer_call_and_return_conditional_losses_3614264�
*heuristic_priority/StatefulPartitionedCallStatefulPartitionedCall3heuristic_detector/StatefulPartitionedCall:output:0heuristic_priority_3616487heuristic_priority_3616489*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_heuristic_priority_layer_call_and_return_conditional_losses_3614281�
&contract_1_5x5/StatefulPartitionedCallStatefulPartitionedCall/expand_1_11x11/StatefulPartitionedCall:output:0contract_1_5x5_3616492contract_1_5x5_3616494*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_1_5x5_layer_call_and_return_conditional_losses_3614298�
concatenate/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0/contract_1_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_3614311�
$expand_2_5x5/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0expand_2_5x5_3616498expand_2_5x5_3616500*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_2_5x5_layer_call_and_return_conditional_losses_3614324�
&contract_2_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_2_5x5/StatefulPartitionedCall:output:0contract_2_3x3_3616503contract_2_3x3_3616505*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_2_3x3_layer_call_and_return_conditional_losses_3614341�
skip_2/PartitionedCallPartitionedCall/contract_2_3x3/StatefulPartitionedCall:output:0/contract_1_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_2_layer_call_and_return_conditional_losses_3614353�
concatenate_1/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_3614362�
$expand_3_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0expand_3_5x5_3616510expand_3_5x5_3616512*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_3_5x5_layer_call_and_return_conditional_losses_3614375�
&contract_3_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_3_5x5/StatefulPartitionedCall:output:0contract_3_3x3_3616515contract_3_3x3_3616517*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_3_3x3_layer_call_and_return_conditional_losses_3614392�
skip_3/PartitionedCallPartitionedCall/contract_3_3x3/StatefulPartitionedCall:output:0skip_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_3_layer_call_and_return_conditional_losses_3614404�
concatenate_2/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_3614413�
$expand_4_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0expand_4_5x5_3616522expand_4_5x5_3616524*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_4_5x5_layer_call_and_return_conditional_losses_3614426�
&contract_4_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_4_5x5/StatefulPartitionedCall:output:0contract_4_3x3_3616527contract_4_3x3_3616529*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_4_3x3_layer_call_and_return_conditional_losses_3614443�
skip_4/PartitionedCallPartitionedCall/contract_4_3x3/StatefulPartitionedCall:output:0skip_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_4_layer_call_and_return_conditional_losses_3614455�
concatenate_3/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_3_layer_call_and_return_conditional_losses_3614464�
$expand_5_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0expand_5_5x5_3616534expand_5_5x5_3616536*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_5_5x5_layer_call_and_return_conditional_losses_3614477�
&contract_5_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_5_5x5/StatefulPartitionedCall:output:0contract_5_3x3_3616539contract_5_3x3_3616541*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_5_3x3_layer_call_and_return_conditional_losses_3614494�
skip_5/PartitionedCallPartitionedCall/contract_5_3x3/StatefulPartitionedCall:output:0skip_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_5_layer_call_and_return_conditional_losses_3614506�
concatenate_4/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_4_layer_call_and_return_conditional_losses_3614515�
$expand_6_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0expand_6_5x5_3616546expand_6_5x5_3616548*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_6_5x5_layer_call_and_return_conditional_losses_3614528�
&contract_6_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_6_5x5/StatefulPartitionedCall:output:0contract_6_3x3_3616551contract_6_3x3_3616553*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_6_3x3_layer_call_and_return_conditional_losses_3614545�
skip_6/PartitionedCallPartitionedCall/contract_6_3x3/StatefulPartitionedCall:output:0skip_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_6_layer_call_and_return_conditional_losses_3614557�
concatenate_5/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_5_layer_call_and_return_conditional_losses_3614566�
$expand_7_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0expand_7_5x5_3616558expand_7_5x5_3616560*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_7_5x5_layer_call_and_return_conditional_losses_3614579�
&contract_7_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_7_5x5/StatefulPartitionedCall:output:0contract_7_3x3_3616563contract_7_3x3_3616565*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_7_3x3_layer_call_and_return_conditional_losses_3614596�
skip_7/PartitionedCallPartitionedCall/contract_7_3x3/StatefulPartitionedCall:output:0skip_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_7_layer_call_and_return_conditional_losses_3614608�
concatenate_6/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_6_layer_call_and_return_conditional_losses_3614617�
$expand_8_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_6/PartitionedCall:output:0expand_8_5x5_3616570expand_8_5x5_3616572*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_8_5x5_layer_call_and_return_conditional_losses_3614630�
&contract_8_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_8_5x5/StatefulPartitionedCall:output:0contract_8_3x3_3616575contract_8_3x3_3616577*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_8_3x3_layer_call_and_return_conditional_losses_3614647�
skip_8/PartitionedCallPartitionedCall/contract_8_3x3/StatefulPartitionedCall:output:0skip_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_8_layer_call_and_return_conditional_losses_3614659�
concatenate_7/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_7_layer_call_and_return_conditional_losses_3614668�
$expand_9_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_7/PartitionedCall:output:0expand_9_5x5_3616582expand_9_5x5_3616584*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_9_5x5_layer_call_and_return_conditional_losses_3614681�
&contract_9_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_9_5x5/StatefulPartitionedCall:output:0contract_9_3x3_3616587contract_9_3x3_3616589*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_9_3x3_layer_call_and_return_conditional_losses_3614698�
skip_9/PartitionedCallPartitionedCall/contract_9_3x3/StatefulPartitionedCall:output:0skip_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_9_layer_call_and_return_conditional_losses_3614710�
concatenate_8/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_8_layer_call_and_return_conditional_losses_3614719�
%expand_10_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_8/PartitionedCall:output:0expand_10_5x5_3616594expand_10_5x5_3616596*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_10_5x5_layer_call_and_return_conditional_losses_3614732�
'contract_10_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_10_5x5/StatefulPartitionedCall:output:0contract_10_3x3_3616599contract_10_3x3_3616601*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_10_3x3_layer_call_and_return_conditional_losses_3614749�
skip_10/PartitionedCallPartitionedCall0contract_10_3x3/StatefulPartitionedCall:output:0skip_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_10_layer_call_and_return_conditional_losses_3614761�
concatenate_9/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_9_layer_call_and_return_conditional_losses_3614770�
%expand_11_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_9/PartitionedCall:output:0expand_11_5x5_3616606expand_11_5x5_3616608*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_11_5x5_layer_call_and_return_conditional_losses_3614783�
'contract_11_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_11_5x5/StatefulPartitionedCall:output:0contract_11_3x3_3616611contract_11_3x3_3616613*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_11_3x3_layer_call_and_return_conditional_losses_3614800�
skip_11/PartitionedCallPartitionedCall0contract_11_3x3/StatefulPartitionedCall:output:0 skip_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_11_layer_call_and_return_conditional_losses_3614812�
concatenate_10/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_10_layer_call_and_return_conditional_losses_3614821�
%expand_12_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_10/PartitionedCall:output:0expand_12_5x5_3616618expand_12_5x5_3616620*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_12_5x5_layer_call_and_return_conditional_losses_3614834�
'contract_12_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_12_5x5/StatefulPartitionedCall:output:0contract_12_3x3_3616623contract_12_3x3_3616625*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_12_3x3_layer_call_and_return_conditional_losses_3614851�
skip_12/PartitionedCallPartitionedCall0contract_12_3x3/StatefulPartitionedCall:output:0 skip_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_12_layer_call_and_return_conditional_losses_3614863�
concatenate_11/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_11_layer_call_and_return_conditional_losses_3614872�
%expand_13_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_11/PartitionedCall:output:0expand_13_5x5_3616630expand_13_5x5_3616632*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_13_5x5_layer_call_and_return_conditional_losses_3614885�
'contract_13_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_13_5x5/StatefulPartitionedCall:output:0contract_13_3x3_3616635contract_13_3x3_3616637*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_13_3x3_layer_call_and_return_conditional_losses_3614902�
skip_13/PartitionedCallPartitionedCall0contract_13_3x3/StatefulPartitionedCall:output:0 skip_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_13_layer_call_and_return_conditional_losses_3614914�
concatenate_12/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_12_layer_call_and_return_conditional_losses_3614923�
%expand_14_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_12/PartitionedCall:output:0expand_14_5x5_3616642expand_14_5x5_3616644*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_14_5x5_layer_call_and_return_conditional_losses_3614936�
'contract_14_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_14_5x5/StatefulPartitionedCall:output:0contract_14_3x3_3616647contract_14_3x3_3616649*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_14_3x3_layer_call_and_return_conditional_losses_3614953�
skip_14/PartitionedCallPartitionedCall0contract_14_3x3/StatefulPartitionedCall:output:0 skip_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_14_layer_call_and_return_conditional_losses_3614965�
concatenate_13/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_14/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_13_layer_call_and_return_conditional_losses_3614974�
%expand_15_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_13/PartitionedCall:output:0expand_15_5x5_3616654expand_15_5x5_3616656*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_15_5x5_layer_call_and_return_conditional_losses_3614987�
'contract_15_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_15_5x5/StatefulPartitionedCall:output:0contract_15_3x3_3616659contract_15_3x3_3616661*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_15_3x3_layer_call_and_return_conditional_losses_3615004�
skip_15/PartitionedCallPartitionedCall0contract_15_3x3/StatefulPartitionedCall:output:0 skip_14/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_15_layer_call_and_return_conditional_losses_3615016�
concatenate_14/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_14_layer_call_and_return_conditional_losses_3615025�
%expand_16_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_14/PartitionedCall:output:0expand_16_5x5_3616666expand_16_5x5_3616668*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_16_5x5_layer_call_and_return_conditional_losses_3615038�
'contract_16_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_16_5x5/StatefulPartitionedCall:output:0contract_16_3x3_3616671contract_16_3x3_3616673*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_16_3x3_layer_call_and_return_conditional_losses_3615055�
skip_16/PartitionedCallPartitionedCall0contract_16_3x3/StatefulPartitionedCall:output:0 skip_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_16_layer_call_and_return_conditional_losses_3615067�
concatenate_15/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_15_layer_call_and_return_conditional_losses_3615076�
%expand_17_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_15/PartitionedCall:output:0expand_17_5x5_3616678expand_17_5x5_3616680*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_17_5x5_layer_call_and_return_conditional_losses_3615089�
'contract_17_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_17_5x5/StatefulPartitionedCall:output:0contract_17_3x3_3616683contract_17_3x3_3616685*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_17_3x3_layer_call_and_return_conditional_losses_3615106�
skip_17/PartitionedCallPartitionedCall0contract_17_3x3/StatefulPartitionedCall:output:0 skip_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_17_layer_call_and_return_conditional_losses_3615118�
concatenate_16/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_16_layer_call_and_return_conditional_losses_3615127�
%expand_18_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_16/PartitionedCall:output:0expand_18_5x5_3616690expand_18_5x5_3616692*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_18_5x5_layer_call_and_return_conditional_losses_3615140�
'contract_18_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_18_5x5/StatefulPartitionedCall:output:0contract_18_3x3_3616695contract_18_3x3_3616697*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_18_3x3_layer_call_and_return_conditional_losses_3615157�
skip_18/PartitionedCallPartitionedCall0contract_18_3x3/StatefulPartitionedCall:output:0 skip_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_18_layer_call_and_return_conditional_losses_3615169�
concatenate_17/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_18/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_17_layer_call_and_return_conditional_losses_3615178�
%expand_19_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_17/PartitionedCall:output:0expand_19_5x5_3616702expand_19_5x5_3616704*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_19_5x5_layer_call_and_return_conditional_losses_3615191�
'contract_19_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_19_5x5/StatefulPartitionedCall:output:0contract_19_3x3_3616707contract_19_3x3_3616709*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_19_3x3_layer_call_and_return_conditional_losses_3615208�
skip_19/PartitionedCallPartitionedCall0contract_19_3x3/StatefulPartitionedCall:output:0 skip_18/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_19_layer_call_and_return_conditional_losses_3615220�
concatenate_18/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_19/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_18_layer_call_and_return_conditional_losses_3615229�
%expand_20_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_18/PartitionedCall:output:0expand_20_5x5_3616714expand_20_5x5_3616716*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_20_5x5_layer_call_and_return_conditional_losses_3615242�
'contract_20_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_20_5x5/StatefulPartitionedCall:output:0contract_20_3x3_3616719contract_20_3x3_3616721*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_20_3x3_layer_call_and_return_conditional_losses_3615259�
skip_20/PartitionedCallPartitionedCall0contract_20_3x3/StatefulPartitionedCall:output:0 skip_19/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_20_layer_call_and_return_conditional_losses_3615271�
all_value_input/PartitionedCallPartitionedCall0contract_20_3x3/StatefulPartitionedCall:output:0$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_all_value_input_layer_call_and_return_conditional_losses_3615280�
)policy_aggregator/StatefulPartitionedCallStatefulPartitionedCall skip_20/PartitionedCall:output:0policy_aggregator_3616726policy_aggregator_3616728*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_policy_aggregator_layer_call_and_return_conditional_losses_3615293�
 flat_value_input/PartitionedCallPartitionedCall(all_value_input/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������,* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_flat_value_input_layer_call_and_return_conditional_losses_3615305�
"border_off/StatefulPartitionedCallStatefulPartitionedCall2policy_aggregator/StatefulPartitionedCall:output:0border_off_3616732border_off_3616734*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_border_off_layer_call_and_return_conditional_losses_3615317^
tf.math.truediv/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �B�
tf.math.truediv/truedivRealDiv)flat_value_input/PartitionedCall:output:0"tf.math.truediv/truediv/y:output:0*
T0*(
_output_shapes
:����������,�
flat_logits/PartitionedCallPartitionedCall+border_off/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_flat_logits_layer_call_and_return_conditional_losses_3615331�
"value_head/StatefulPartitionedCallStatefulPartitionedCalltf.math.truediv/truediv:z:0value_head_3616740value_head_3616742*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_value_head_layer_call_and_return_conditional_losses_3615344�
policy_head/PartitionedCallPartitionedCall$flat_logits/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_policy_head_layer_call_and_return_conditional_losses_3615355t
IdentityIdentity$policy_head/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������|

Identity_1Identity+value_head/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp#^border_off/StatefulPartitionedCall(^contract_10_3x3/StatefulPartitionedCall(^contract_11_3x3/StatefulPartitionedCall(^contract_12_3x3/StatefulPartitionedCall(^contract_13_3x3/StatefulPartitionedCall(^contract_14_3x3/StatefulPartitionedCall(^contract_15_3x3/StatefulPartitionedCall(^contract_16_3x3/StatefulPartitionedCall(^contract_17_3x3/StatefulPartitionedCall(^contract_18_3x3/StatefulPartitionedCall(^contract_19_3x3/StatefulPartitionedCall'^contract_1_5x5/StatefulPartitionedCall(^contract_20_3x3/StatefulPartitionedCall'^contract_2_3x3/StatefulPartitionedCall'^contract_3_3x3/StatefulPartitionedCall'^contract_4_3x3/StatefulPartitionedCall'^contract_5_3x3/StatefulPartitionedCall'^contract_6_3x3/StatefulPartitionedCall'^contract_7_3x3/StatefulPartitionedCall'^contract_8_3x3/StatefulPartitionedCall'^contract_9_3x3/StatefulPartitionedCall&^expand_10_5x5/StatefulPartitionedCall&^expand_11_5x5/StatefulPartitionedCall&^expand_12_5x5/StatefulPartitionedCall&^expand_13_5x5/StatefulPartitionedCall&^expand_14_5x5/StatefulPartitionedCall&^expand_15_5x5/StatefulPartitionedCall&^expand_16_5x5/StatefulPartitionedCall&^expand_17_5x5/StatefulPartitionedCall&^expand_18_5x5/StatefulPartitionedCall&^expand_19_5x5/StatefulPartitionedCall'^expand_1_11x11/StatefulPartitionedCall&^expand_20_5x5/StatefulPartitionedCall%^expand_2_5x5/StatefulPartitionedCall%^expand_3_5x5/StatefulPartitionedCall%^expand_4_5x5/StatefulPartitionedCall%^expand_5_5x5/StatefulPartitionedCall%^expand_6_5x5/StatefulPartitionedCall%^expand_7_5x5/StatefulPartitionedCall%^expand_8_5x5/StatefulPartitionedCall%^expand_9_5x5/StatefulPartitionedCall+^heuristic_detector/StatefulPartitionedCall+^heuristic_priority/StatefulPartitionedCall*^policy_aggregator/StatefulPartitionedCall#^value_head/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
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
:���������
 
_user_specified_nameinputs
�
v
L__inference_all_value_input_layer_call_and_return_conditional_losses_3615280

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
:���������_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������	:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
T
(__inference_skip_4_layer_call_fn_3618126
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_4_layer_call_and_return_conditional_losses_3614455h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
L__inference_contract_15_3x3_layer_call_and_return_conditional_losses_3618835

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
o
C__inference_skip_2_layer_call_and_return_conditional_losses_3618002
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
L__inference_contract_11_3x3_layer_call_and_return_conditional_losses_3618575

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
K__inference_contract_9_3x3_layer_call_and_return_conditional_losses_3618445

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
t
J__inference_concatenate_4_layer_call_and_return_conditional_losses_3614515

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
:���������	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
J__inference_expand_11_5x5_layer_call_and_return_conditional_losses_3618555

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:��������� m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
/__inference_expand_12_5x5_layer_call_fn_3618609

inputs!
unknown:	 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_12_5x5_layer_call_and_return_conditional_losses_3614834w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
I__inference_expand_5_5x5_layer_call_and_return_conditional_losses_3614477

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:��������� m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
T
(__inference_skip_6_layer_call_fn_3618256
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_6_layer_call_and_return_conditional_losses_3614557h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
n
D__inference_skip_16_layer_call_and_return_conditional_losses_3615067

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
[
/__inference_concatenate_3_layer_call_fn_3618138
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_3_layer_call_and_return_conditional_losses_3614464h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
m
C__inference_skip_6_layer_call_and_return_conditional_losses_3614557

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
L__inference_contract_13_3x3_layer_call_and_return_conditional_losses_3614902

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
T
(__inference_skip_8_layer_call_fn_3618386
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_8_layer_call_and_return_conditional_losses_3614659h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
\
0__inference_concatenate_16_layer_call_fn_3618983
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_16_layer_call_and_return_conditional_losses_3615127h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
p
D__inference_skip_10_layer_call_and_return_conditional_losses_3618522
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
v
J__inference_concatenate_4_layer_call_and_return_conditional_losses_3618210
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
:���������	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
J__inference_expand_16_5x5_layer_call_and_return_conditional_losses_3618880

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:��������� m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
[
/__inference_concatenate_5_layer_call_fn_3618268
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_5_layer_call_and_return_conditional_losses_3614566h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
/__inference_expand_17_5x5_layer_call_fn_3618934

inputs!
unknown:	 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_17_5x5_layer_call_and_return_conditional_losses_3615089w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
t
J__inference_concatenate_2_layer_call_and_return_conditional_losses_3614413

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
:���������	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
.__inference_expand_8_5x5_layer_call_fn_3618349

inputs!
unknown:	 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_8_5x5_layer_call_and_return_conditional_losses_3614630w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
p
D__inference_skip_14_layer_call_and_return_conditional_losses_3618782
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
\
0__inference_concatenate_12_layer_call_fn_3618723
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_12_layer_call_and_return_conditional_losses_3614923h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
m
C__inference_skip_7_layer_call_and_return_conditional_losses_3614608

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
.__inference_expand_6_5x5_layer_call_fn_3618219

inputs!
unknown:	 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_6_5x5_layer_call_and_return_conditional_losses_3614528w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
w
K__inference_concatenate_11_layer_call_and_return_conditional_losses_3618665
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
:���������	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
0__inference_contract_2_3x3_layer_call_fn_3617979

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_2_3x3_layer_call_and_return_conditional_losses_3614341w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
m
C__inference_skip_8_layer_call_and_return_conditional_losses_3614659

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
J__inference_expand_11_5x5_layer_call_and_return_conditional_losses_3614783

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:��������� m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
v
J__inference_concatenate_6_layer_call_and_return_conditional_losses_3618340
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
:���������	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
K__inference_contract_1_5x5_layer_call_and_return_conditional_losses_3614298

inputs9
conv2d_readvariableop_resource:�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
4__inference_heuristic_detector_layer_call_fn_3617866

inputs"
unknown:�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_heuristic_detector_layer_call_and_return_conditional_losses_3614264x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
T
(__inference_skip_9_layer_call_fn_3618451
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_9_layer_call_and_return_conditional_losses_3614710h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
L__inference_contract_20_3x3_layer_call_and_return_conditional_losses_3619160

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
1__inference_contract_12_3x3_layer_call_fn_3618629

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_12_3x3_layer_call_and_return_conditional_losses_3614851w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
Ё
�-
J__inference_gomoku_resnet_layer_call_and_return_conditional_losses_3617394

inputs1
expand_1_11x11_3617123:�%
expand_1_11x11_3617125:	�5
heuristic_detector_3617128:�)
heuristic_detector_3617130:	�5
heuristic_priority_3617133:�(
heuristic_priority_3617135:1
contract_1_5x5_3617138:�$
contract_1_5x5_3617140:.
expand_2_5x5_3617144:	 "
expand_2_5x5_3617146: 0
contract_2_3x3_3617149: $
contract_2_3x3_3617151:.
expand_3_5x5_3617156:	 "
expand_3_5x5_3617158: 0
contract_3_3x3_3617161: $
contract_3_3x3_3617163:.
expand_4_5x5_3617168:	 "
expand_4_5x5_3617170: 0
contract_4_3x3_3617173: $
contract_4_3x3_3617175:.
expand_5_5x5_3617180:	 "
expand_5_5x5_3617182: 0
contract_5_3x3_3617185: $
contract_5_3x3_3617187:.
expand_6_5x5_3617192:	 "
expand_6_5x5_3617194: 0
contract_6_3x3_3617197: $
contract_6_3x3_3617199:.
expand_7_5x5_3617204:	 "
expand_7_5x5_3617206: 0
contract_7_3x3_3617209: $
contract_7_3x3_3617211:.
expand_8_5x5_3617216:	 "
expand_8_5x5_3617218: 0
contract_8_3x3_3617221: $
contract_8_3x3_3617223:.
expand_9_5x5_3617228:	 "
expand_9_5x5_3617230: 0
contract_9_3x3_3617233: $
contract_9_3x3_3617235:/
expand_10_5x5_3617240:	 #
expand_10_5x5_3617242: 1
contract_10_3x3_3617245: %
contract_10_3x3_3617247:/
expand_11_5x5_3617252:	 #
expand_11_5x5_3617254: 1
contract_11_3x3_3617257: %
contract_11_3x3_3617259:/
expand_12_5x5_3617264:	 #
expand_12_5x5_3617266: 1
contract_12_3x3_3617269: %
contract_12_3x3_3617271:/
expand_13_5x5_3617276:	 #
expand_13_5x5_3617278: 1
contract_13_3x3_3617281: %
contract_13_3x3_3617283:/
expand_14_5x5_3617288:	 #
expand_14_5x5_3617290: 1
contract_14_3x3_3617293: %
contract_14_3x3_3617295:/
expand_15_5x5_3617300:	 #
expand_15_5x5_3617302: 1
contract_15_3x3_3617305: %
contract_15_3x3_3617307:/
expand_16_5x5_3617312:	 #
expand_16_5x5_3617314: 1
contract_16_3x3_3617317: %
contract_16_3x3_3617319:/
expand_17_5x5_3617324:	 #
expand_17_5x5_3617326: 1
contract_17_3x3_3617329: %
contract_17_3x3_3617331:/
expand_18_5x5_3617336:	 #
expand_18_5x5_3617338: 1
contract_18_3x3_3617341: %
contract_18_3x3_3617343:/
expand_19_5x5_3617348:	 #
expand_19_5x5_3617350: 1
contract_19_3x3_3617353: %
contract_19_3x3_3617355:/
expand_20_5x5_3617360:	 #
expand_20_5x5_3617362: 1
contract_20_3x3_3617365: %
contract_20_3x3_3617367:3
policy_aggregator_3617372:'
policy_aggregator_3617374:,
border_off_3617378: 
border_off_3617380:%
value_head_3617386:	�, 
value_head_3617388:
identity

identity_1��"border_off/StatefulPartitionedCall�'contract_10_3x3/StatefulPartitionedCall�'contract_11_3x3/StatefulPartitionedCall�'contract_12_3x3/StatefulPartitionedCall�'contract_13_3x3/StatefulPartitionedCall�'contract_14_3x3/StatefulPartitionedCall�'contract_15_3x3/StatefulPartitionedCall�'contract_16_3x3/StatefulPartitionedCall�'contract_17_3x3/StatefulPartitionedCall�'contract_18_3x3/StatefulPartitionedCall�'contract_19_3x3/StatefulPartitionedCall�&contract_1_5x5/StatefulPartitionedCall�'contract_20_3x3/StatefulPartitionedCall�&contract_2_3x3/StatefulPartitionedCall�&contract_3_3x3/StatefulPartitionedCall�&contract_4_3x3/StatefulPartitionedCall�&contract_5_3x3/StatefulPartitionedCall�&contract_6_3x3/StatefulPartitionedCall�&contract_7_3x3/StatefulPartitionedCall�&contract_8_3x3/StatefulPartitionedCall�&contract_9_3x3/StatefulPartitionedCall�%expand_10_5x5/StatefulPartitionedCall�%expand_11_5x5/StatefulPartitionedCall�%expand_12_5x5/StatefulPartitionedCall�%expand_13_5x5/StatefulPartitionedCall�%expand_14_5x5/StatefulPartitionedCall�%expand_15_5x5/StatefulPartitionedCall�%expand_16_5x5/StatefulPartitionedCall�%expand_17_5x5/StatefulPartitionedCall�%expand_18_5x5/StatefulPartitionedCall�%expand_19_5x5/StatefulPartitionedCall�&expand_1_11x11/StatefulPartitionedCall�%expand_20_5x5/StatefulPartitionedCall�$expand_2_5x5/StatefulPartitionedCall�$expand_3_5x5/StatefulPartitionedCall�$expand_4_5x5/StatefulPartitionedCall�$expand_5_5x5/StatefulPartitionedCall�$expand_6_5x5/StatefulPartitionedCall�$expand_7_5x5/StatefulPartitionedCall�$expand_8_5x5/StatefulPartitionedCall�$expand_9_5x5/StatefulPartitionedCall�*heuristic_detector/StatefulPartitionedCall�*heuristic_priority/StatefulPartitionedCall�)policy_aggregator/StatefulPartitionedCall�"value_head/StatefulPartitionedCall�
&expand_1_11x11/StatefulPartitionedCallStatefulPartitionedCallinputsexpand_1_11x11_3617123expand_1_11x11_3617125*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_expand_1_11x11_layer_call_and_return_conditional_losses_3614247�
*heuristic_detector/StatefulPartitionedCallStatefulPartitionedCallinputsheuristic_detector_3617128heuristic_detector_3617130*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_heuristic_detector_layer_call_and_return_conditional_losses_3614264�
*heuristic_priority/StatefulPartitionedCallStatefulPartitionedCall3heuristic_detector/StatefulPartitionedCall:output:0heuristic_priority_3617133heuristic_priority_3617135*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_heuristic_priority_layer_call_and_return_conditional_losses_3614281�
&contract_1_5x5/StatefulPartitionedCallStatefulPartitionedCall/expand_1_11x11/StatefulPartitionedCall:output:0contract_1_5x5_3617138contract_1_5x5_3617140*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_1_5x5_layer_call_and_return_conditional_losses_3614298�
concatenate/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0/contract_1_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_3614311�
$expand_2_5x5/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0expand_2_5x5_3617144expand_2_5x5_3617146*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_2_5x5_layer_call_and_return_conditional_losses_3614324�
&contract_2_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_2_5x5/StatefulPartitionedCall:output:0contract_2_3x3_3617149contract_2_3x3_3617151*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_2_3x3_layer_call_and_return_conditional_losses_3614341�
skip_2/PartitionedCallPartitionedCall/contract_2_3x3/StatefulPartitionedCall:output:0/contract_1_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_2_layer_call_and_return_conditional_losses_3614353�
concatenate_1/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_3614362�
$expand_3_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0expand_3_5x5_3617156expand_3_5x5_3617158*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_3_5x5_layer_call_and_return_conditional_losses_3614375�
&contract_3_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_3_5x5/StatefulPartitionedCall:output:0contract_3_3x3_3617161contract_3_3x3_3617163*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_3_3x3_layer_call_and_return_conditional_losses_3614392�
skip_3/PartitionedCallPartitionedCall/contract_3_3x3/StatefulPartitionedCall:output:0skip_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_3_layer_call_and_return_conditional_losses_3614404�
concatenate_2/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_3614413�
$expand_4_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0expand_4_5x5_3617168expand_4_5x5_3617170*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_4_5x5_layer_call_and_return_conditional_losses_3614426�
&contract_4_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_4_5x5/StatefulPartitionedCall:output:0contract_4_3x3_3617173contract_4_3x3_3617175*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_4_3x3_layer_call_and_return_conditional_losses_3614443�
skip_4/PartitionedCallPartitionedCall/contract_4_3x3/StatefulPartitionedCall:output:0skip_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_4_layer_call_and_return_conditional_losses_3614455�
concatenate_3/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_3_layer_call_and_return_conditional_losses_3614464�
$expand_5_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0expand_5_5x5_3617180expand_5_5x5_3617182*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_5_5x5_layer_call_and_return_conditional_losses_3614477�
&contract_5_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_5_5x5/StatefulPartitionedCall:output:0contract_5_3x3_3617185contract_5_3x3_3617187*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_5_3x3_layer_call_and_return_conditional_losses_3614494�
skip_5/PartitionedCallPartitionedCall/contract_5_3x3/StatefulPartitionedCall:output:0skip_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_5_layer_call_and_return_conditional_losses_3614506�
concatenate_4/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_4_layer_call_and_return_conditional_losses_3614515�
$expand_6_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0expand_6_5x5_3617192expand_6_5x5_3617194*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_6_5x5_layer_call_and_return_conditional_losses_3614528�
&contract_6_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_6_5x5/StatefulPartitionedCall:output:0contract_6_3x3_3617197contract_6_3x3_3617199*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_6_3x3_layer_call_and_return_conditional_losses_3614545�
skip_6/PartitionedCallPartitionedCall/contract_6_3x3/StatefulPartitionedCall:output:0skip_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_6_layer_call_and_return_conditional_losses_3614557�
concatenate_5/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_5_layer_call_and_return_conditional_losses_3614566�
$expand_7_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0expand_7_5x5_3617204expand_7_5x5_3617206*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_7_5x5_layer_call_and_return_conditional_losses_3614579�
&contract_7_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_7_5x5/StatefulPartitionedCall:output:0contract_7_3x3_3617209contract_7_3x3_3617211*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_7_3x3_layer_call_and_return_conditional_losses_3614596�
skip_7/PartitionedCallPartitionedCall/contract_7_3x3/StatefulPartitionedCall:output:0skip_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_7_layer_call_and_return_conditional_losses_3614608�
concatenate_6/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_6_layer_call_and_return_conditional_losses_3614617�
$expand_8_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_6/PartitionedCall:output:0expand_8_5x5_3617216expand_8_5x5_3617218*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_8_5x5_layer_call_and_return_conditional_losses_3614630�
&contract_8_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_8_5x5/StatefulPartitionedCall:output:0contract_8_3x3_3617221contract_8_3x3_3617223*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_8_3x3_layer_call_and_return_conditional_losses_3614647�
skip_8/PartitionedCallPartitionedCall/contract_8_3x3/StatefulPartitionedCall:output:0skip_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_8_layer_call_and_return_conditional_losses_3614659�
concatenate_7/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_7_layer_call_and_return_conditional_losses_3614668�
$expand_9_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_7/PartitionedCall:output:0expand_9_5x5_3617228expand_9_5x5_3617230*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_9_5x5_layer_call_and_return_conditional_losses_3614681�
&contract_9_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_9_5x5/StatefulPartitionedCall:output:0contract_9_3x3_3617233contract_9_3x3_3617235*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_9_3x3_layer_call_and_return_conditional_losses_3614698�
skip_9/PartitionedCallPartitionedCall/contract_9_3x3/StatefulPartitionedCall:output:0skip_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_9_layer_call_and_return_conditional_losses_3614710�
concatenate_8/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_8_layer_call_and_return_conditional_losses_3614719�
%expand_10_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_8/PartitionedCall:output:0expand_10_5x5_3617240expand_10_5x5_3617242*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_10_5x5_layer_call_and_return_conditional_losses_3614732�
'contract_10_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_10_5x5/StatefulPartitionedCall:output:0contract_10_3x3_3617245contract_10_3x3_3617247*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_10_3x3_layer_call_and_return_conditional_losses_3614749�
skip_10/PartitionedCallPartitionedCall0contract_10_3x3/StatefulPartitionedCall:output:0skip_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_10_layer_call_and_return_conditional_losses_3614761�
concatenate_9/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_9_layer_call_and_return_conditional_losses_3614770�
%expand_11_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_9/PartitionedCall:output:0expand_11_5x5_3617252expand_11_5x5_3617254*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_11_5x5_layer_call_and_return_conditional_losses_3614783�
'contract_11_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_11_5x5/StatefulPartitionedCall:output:0contract_11_3x3_3617257contract_11_3x3_3617259*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_11_3x3_layer_call_and_return_conditional_losses_3614800�
skip_11/PartitionedCallPartitionedCall0contract_11_3x3/StatefulPartitionedCall:output:0 skip_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_11_layer_call_and_return_conditional_losses_3614812�
concatenate_10/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_10_layer_call_and_return_conditional_losses_3614821�
%expand_12_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_10/PartitionedCall:output:0expand_12_5x5_3617264expand_12_5x5_3617266*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_12_5x5_layer_call_and_return_conditional_losses_3614834�
'contract_12_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_12_5x5/StatefulPartitionedCall:output:0contract_12_3x3_3617269contract_12_3x3_3617271*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_12_3x3_layer_call_and_return_conditional_losses_3614851�
skip_12/PartitionedCallPartitionedCall0contract_12_3x3/StatefulPartitionedCall:output:0 skip_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_12_layer_call_and_return_conditional_losses_3614863�
concatenate_11/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_11_layer_call_and_return_conditional_losses_3614872�
%expand_13_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_11/PartitionedCall:output:0expand_13_5x5_3617276expand_13_5x5_3617278*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_13_5x5_layer_call_and_return_conditional_losses_3614885�
'contract_13_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_13_5x5/StatefulPartitionedCall:output:0contract_13_3x3_3617281contract_13_3x3_3617283*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_13_3x3_layer_call_and_return_conditional_losses_3614902�
skip_13/PartitionedCallPartitionedCall0contract_13_3x3/StatefulPartitionedCall:output:0 skip_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_13_layer_call_and_return_conditional_losses_3614914�
concatenate_12/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_12_layer_call_and_return_conditional_losses_3614923�
%expand_14_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_12/PartitionedCall:output:0expand_14_5x5_3617288expand_14_5x5_3617290*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_14_5x5_layer_call_and_return_conditional_losses_3614936�
'contract_14_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_14_5x5/StatefulPartitionedCall:output:0contract_14_3x3_3617293contract_14_3x3_3617295*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_14_3x3_layer_call_and_return_conditional_losses_3614953�
skip_14/PartitionedCallPartitionedCall0contract_14_3x3/StatefulPartitionedCall:output:0 skip_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_14_layer_call_and_return_conditional_losses_3614965�
concatenate_13/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_14/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_13_layer_call_and_return_conditional_losses_3614974�
%expand_15_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_13/PartitionedCall:output:0expand_15_5x5_3617300expand_15_5x5_3617302*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_15_5x5_layer_call_and_return_conditional_losses_3614987�
'contract_15_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_15_5x5/StatefulPartitionedCall:output:0contract_15_3x3_3617305contract_15_3x3_3617307*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_15_3x3_layer_call_and_return_conditional_losses_3615004�
skip_15/PartitionedCallPartitionedCall0contract_15_3x3/StatefulPartitionedCall:output:0 skip_14/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_15_layer_call_and_return_conditional_losses_3615016�
concatenate_14/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_14_layer_call_and_return_conditional_losses_3615025�
%expand_16_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_14/PartitionedCall:output:0expand_16_5x5_3617312expand_16_5x5_3617314*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_16_5x5_layer_call_and_return_conditional_losses_3615038�
'contract_16_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_16_5x5/StatefulPartitionedCall:output:0contract_16_3x3_3617317contract_16_3x3_3617319*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_16_3x3_layer_call_and_return_conditional_losses_3615055�
skip_16/PartitionedCallPartitionedCall0contract_16_3x3/StatefulPartitionedCall:output:0 skip_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_16_layer_call_and_return_conditional_losses_3615067�
concatenate_15/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_15_layer_call_and_return_conditional_losses_3615076�
%expand_17_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_15/PartitionedCall:output:0expand_17_5x5_3617324expand_17_5x5_3617326*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_17_5x5_layer_call_and_return_conditional_losses_3615089�
'contract_17_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_17_5x5/StatefulPartitionedCall:output:0contract_17_3x3_3617329contract_17_3x3_3617331*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_17_3x3_layer_call_and_return_conditional_losses_3615106�
skip_17/PartitionedCallPartitionedCall0contract_17_3x3/StatefulPartitionedCall:output:0 skip_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_17_layer_call_and_return_conditional_losses_3615118�
concatenate_16/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_16_layer_call_and_return_conditional_losses_3615127�
%expand_18_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_16/PartitionedCall:output:0expand_18_5x5_3617336expand_18_5x5_3617338*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_18_5x5_layer_call_and_return_conditional_losses_3615140�
'contract_18_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_18_5x5/StatefulPartitionedCall:output:0contract_18_3x3_3617341contract_18_3x3_3617343*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_18_3x3_layer_call_and_return_conditional_losses_3615157�
skip_18/PartitionedCallPartitionedCall0contract_18_3x3/StatefulPartitionedCall:output:0 skip_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_18_layer_call_and_return_conditional_losses_3615169�
concatenate_17/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_18/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_17_layer_call_and_return_conditional_losses_3615178�
%expand_19_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_17/PartitionedCall:output:0expand_19_5x5_3617348expand_19_5x5_3617350*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_19_5x5_layer_call_and_return_conditional_losses_3615191�
'contract_19_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_19_5x5/StatefulPartitionedCall:output:0contract_19_3x3_3617353contract_19_3x3_3617355*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_19_3x3_layer_call_and_return_conditional_losses_3615208�
skip_19/PartitionedCallPartitionedCall0contract_19_3x3/StatefulPartitionedCall:output:0 skip_18/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_19_layer_call_and_return_conditional_losses_3615220�
concatenate_18/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_19/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_18_layer_call_and_return_conditional_losses_3615229�
%expand_20_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_18/PartitionedCall:output:0expand_20_5x5_3617360expand_20_5x5_3617362*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_20_5x5_layer_call_and_return_conditional_losses_3615242�
'contract_20_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_20_5x5/StatefulPartitionedCall:output:0contract_20_3x3_3617365contract_20_3x3_3617367*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_20_3x3_layer_call_and_return_conditional_losses_3615259�
skip_20/PartitionedCallPartitionedCall0contract_20_3x3/StatefulPartitionedCall:output:0 skip_19/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_20_layer_call_and_return_conditional_losses_3615271�
all_value_input/PartitionedCallPartitionedCall0contract_20_3x3/StatefulPartitionedCall:output:0$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_all_value_input_layer_call_and_return_conditional_losses_3615280�
)policy_aggregator/StatefulPartitionedCallStatefulPartitionedCall skip_20/PartitionedCall:output:0policy_aggregator_3617372policy_aggregator_3617374*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_policy_aggregator_layer_call_and_return_conditional_losses_3615293�
 flat_value_input/PartitionedCallPartitionedCall(all_value_input/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������,* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_flat_value_input_layer_call_and_return_conditional_losses_3615305�
"border_off/StatefulPartitionedCallStatefulPartitionedCall2policy_aggregator/StatefulPartitionedCall:output:0border_off_3617378border_off_3617380*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_border_off_layer_call_and_return_conditional_losses_3615317^
tf.math.truediv/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �B�
tf.math.truediv/truedivRealDiv)flat_value_input/PartitionedCall:output:0"tf.math.truediv/truediv/y:output:0*
T0*(
_output_shapes
:����������,�
flat_logits/PartitionedCallPartitionedCall+border_off/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_flat_logits_layer_call_and_return_conditional_losses_3615331�
"value_head/StatefulPartitionedCallStatefulPartitionedCalltf.math.truediv/truediv:z:0value_head_3617386value_head_3617388*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_value_head_layer_call_and_return_conditional_losses_3615344�
policy_head/PartitionedCallPartitionedCall$flat_logits/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_policy_head_layer_call_and_return_conditional_losses_3615355t
IdentityIdentity$policy_head/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������|

Identity_1Identity+value_head/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp#^border_off/StatefulPartitionedCall(^contract_10_3x3/StatefulPartitionedCall(^contract_11_3x3/StatefulPartitionedCall(^contract_12_3x3/StatefulPartitionedCall(^contract_13_3x3/StatefulPartitionedCall(^contract_14_3x3/StatefulPartitionedCall(^contract_15_3x3/StatefulPartitionedCall(^contract_16_3x3/StatefulPartitionedCall(^contract_17_3x3/StatefulPartitionedCall(^contract_18_3x3/StatefulPartitionedCall(^contract_19_3x3/StatefulPartitionedCall'^contract_1_5x5/StatefulPartitionedCall(^contract_20_3x3/StatefulPartitionedCall'^contract_2_3x3/StatefulPartitionedCall'^contract_3_3x3/StatefulPartitionedCall'^contract_4_3x3/StatefulPartitionedCall'^contract_5_3x3/StatefulPartitionedCall'^contract_6_3x3/StatefulPartitionedCall'^contract_7_3x3/StatefulPartitionedCall'^contract_8_3x3/StatefulPartitionedCall'^contract_9_3x3/StatefulPartitionedCall&^expand_10_5x5/StatefulPartitionedCall&^expand_11_5x5/StatefulPartitionedCall&^expand_12_5x5/StatefulPartitionedCall&^expand_13_5x5/StatefulPartitionedCall&^expand_14_5x5/StatefulPartitionedCall&^expand_15_5x5/StatefulPartitionedCall&^expand_16_5x5/StatefulPartitionedCall&^expand_17_5x5/StatefulPartitionedCall&^expand_18_5x5/StatefulPartitionedCall&^expand_19_5x5/StatefulPartitionedCall'^expand_1_11x11/StatefulPartitionedCall&^expand_20_5x5/StatefulPartitionedCall%^expand_2_5x5/StatefulPartitionedCall%^expand_3_5x5/StatefulPartitionedCall%^expand_4_5x5/StatefulPartitionedCall%^expand_5_5x5/StatefulPartitionedCall%^expand_6_5x5/StatefulPartitionedCall%^expand_7_5x5/StatefulPartitionedCall%^expand_8_5x5/StatefulPartitionedCall%^expand_9_5x5/StatefulPartitionedCall+^heuristic_detector/StatefulPartitionedCall+^heuristic_priority/StatefulPartitionedCall*^policy_aggregator/StatefulPartitionedCall#^value_head/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
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
:���������
 
_user_specified_nameinputs
�
p
D__inference_skip_18_layer_call_and_return_conditional_losses_3619042
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
n
D__inference_skip_15_layer_call_and_return_conditional_losses_3615016

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
U
)__inference_skip_18_layer_call_fn_3619036
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_18_layer_call_and_return_conditional_losses_3615169h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
t
J__inference_concatenate_8_layer_call_and_return_conditional_losses_3614719

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
:���������	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
K__inference_contract_7_3x3_layer_call_and_return_conditional_losses_3614596

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
w
K__inference_concatenate_10_layer_call_and_return_conditional_losses_3618600
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
:���������	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
n
D__inference_skip_13_layer_call_and_return_conditional_losses_3614914

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
n
D__inference_skip_11_layer_call_and_return_conditional_losses_3614812

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
U
)__inference_skip_15_layer_call_fn_3618841
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_15_layer_call_and_return_conditional_losses_3615016h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
u
K__inference_concatenate_18_layer_call_and_return_conditional_losses_3615229

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
:���������	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
O__inference_heuristic_detector_layer_call_and_return_conditional_losses_3617877

inputs9
conv2d_readvariableop_resource:�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
/__inference_expand_13_5x5_layer_call_fn_3618674

inputs!
unknown:	 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_13_5x5_layer_call_and_return_conditional_losses_3614885w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
1__inference_contract_18_3x3_layer_call_fn_3619019

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_18_3x3_layer_call_and_return_conditional_losses_3615157w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
I__inference_expand_4_5x5_layer_call_and_return_conditional_losses_3618100

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:��������� m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
T
(__inference_skip_2_layer_call_fn_3617996
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_2_layer_call_and_return_conditional_losses_3614353h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
t
J__inference_concatenate_1_layer_call_and_return_conditional_losses_3614362

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
:���������	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
0__inference_expand_1_11x11_layer_call_fn_3617886

inputs"
unknown:�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_expand_1_11x11_layer_call_and_return_conditional_losses_3614247x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
I
-__inference_flat_logits_layer_call_fn_3619240

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_flat_logits_layer_call_and_return_conditional_losses_3615331a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
4__inference_heuristic_priority_layer_call_fn_3617906

inputs"
unknown:�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_heuristic_priority_layer_call_and_return_conditional_losses_3614281w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
w
K__inference_concatenate_18_layer_call_and_return_conditional_losses_3619120
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
:���������	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
I__inference_expand_4_5x5_layer_call_and_return_conditional_losses_3614426

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:��������� m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
/__inference_expand_18_5x5_layer_call_fn_3618999

inputs!
unknown:	 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_18_5x5_layer_call_and_return_conditional_losses_3615140w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
p
D__inference_skip_19_layer_call_and_return_conditional_losses_3619107
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
.__inference_expand_7_5x5_layer_call_fn_3618284

inputs!
unknown:	 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_7_5x5_layer_call_and_return_conditional_losses_3614579w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
u
K__inference_concatenate_14_layer_call_and_return_conditional_losses_3615025

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
:���������	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
G__inference_value_head_layer_call_and_return_conditional_losses_3615344

inputs1
matmul_readvariableop_resource:	�,-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�,*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������,: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������,
 
_user_specified_nameinputs
�
�
L__inference_contract_18_3x3_layer_call_and_return_conditional_losses_3619030

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
G__inference_border_off_layer_call_and_return_conditional_losses_3615317

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
v
J__inference_concatenate_8_layer_call_and_return_conditional_losses_3618470
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
:���������	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
K__inference_contract_6_3x3_layer_call_and_return_conditional_losses_3614545

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
J__inference_expand_19_5x5_layer_call_and_return_conditional_losses_3619075

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:��������� m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
.__inference_expand_2_5x5_layer_call_fn_3617959

inputs!
unknown:	 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_2_5x5_layer_call_and_return_conditional_losses_3614324w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
K__inference_contract_4_3x3_layer_call_and_return_conditional_losses_3618120

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
J__inference_expand_20_5x5_layer_call_and_return_conditional_losses_3619140

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:��������� m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
J__inference_expand_19_5x5_layer_call_and_return_conditional_losses_3615191

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:��������� m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
O__inference_heuristic_priority_layer_call_and_return_conditional_losses_3617917

inputs9
conv2d_readvariableop_resource:�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������X
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:���������_
IdentityIdentityTanh:y:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
J__inference_expand_13_5x5_layer_call_and_return_conditional_losses_3614885

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:��������� m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
K__inference_expand_1_11x11_layer_call_and_return_conditional_losses_3614247

inputs9
conv2d_readvariableop_resource:�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������a
SoftplusSoftplusBiasAdd:output:0*
T0*0
_output_shapes
:����������n
IdentityIdentitySoftplus:activations:0^NoOp*
T0*0
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
J__inference_expand_14_5x5_layer_call_and_return_conditional_losses_3618750

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:��������� m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
o
C__inference_skip_8_layer_call_and_return_conditional_losses_3618392
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
t
J__inference_concatenate_9_layer_call_and_return_conditional_losses_3614770

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
:���������	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
J__inference_expand_17_5x5_layer_call_and_return_conditional_losses_3618945

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:��������� m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
K__inference_contract_2_3x3_layer_call_and_return_conditional_losses_3614341

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
n
D__inference_skip_14_layer_call_and_return_conditional_losses_3614965

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs
�*
�
/__inference_gomoku_resnet_layer_call_fn_3617120

inputs"
unknown:�
	unknown_0:	�$
	unknown_1:�
	unknown_2:	�$
	unknown_3:�
	unknown_4:$
	unknown_5:�
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

unknown_87:	�,

unknown_88:
identity

identity_1��StatefulPartitionedCall�
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
':����������:���������*|
_read_only_resource_inputs^
\Z	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gomoku_resnet_layer_call_and_return_conditional_losses_3616748p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
v
J__inference_concatenate_2_layer_call_and_return_conditional_losses_3618080
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
:���������	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
K__inference_contract_6_3x3_layer_call_and_return_conditional_losses_3618250

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
v
J__inference_concatenate_7_layer_call_and_return_conditional_losses_3618405
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
:���������	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
U
)__inference_skip_12_layer_call_fn_3618646
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_12_layer_call_and_return_conditional_losses_3614863h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
t
J__inference_concatenate_3_layer_call_and_return_conditional_losses_3614464

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
:���������	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
L__inference_contract_10_3x3_layer_call_and_return_conditional_losses_3614749

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
\
0__inference_concatenate_14_layer_call_fn_3618853
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_14_layer_call_and_return_conditional_losses_3615025h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�

�
G__inference_border_off_layer_call_and_return_conditional_losses_3619224

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
n
D__inference_skip_18_layer_call_and_return_conditional_losses_3615169

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
I
-__inference_policy_head_layer_call_fn_3619251

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_policy_head_layer_call_and_return_conditional_losses_3615355a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
0__inference_contract_3_3x3_layer_call_fn_3618044

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_3_3x3_layer_call_and_return_conditional_losses_3614392w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
I__inference_expand_3_5x5_layer_call_and_return_conditional_losses_3614375

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:��������� m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
0__inference_contract_7_3x3_layer_call_fn_3618304

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_7_3x3_layer_call_and_return_conditional_losses_3614596w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
N__inference_policy_aggregator_layer_call_and_return_conditional_losses_3615293

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
\
0__inference_concatenate_11_layer_call_fn_3618658
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_11_layer_call_and_return_conditional_losses_3614872h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
o
C__inference_skip_3_layer_call_and_return_conditional_losses_3618067
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
d
H__inference_flat_logits_layer_call_and_return_conditional_losses_3615331

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����i  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
u
K__inference_concatenate_17_layer_call_and_return_conditional_losses_3615178

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
:���������	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
o
C__inference_skip_4_layer_call_and_return_conditional_losses_3618132
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
I__inference_expand_2_5x5_layer_call_and_return_conditional_losses_3617970

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:��������� m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
w
K__inference_concatenate_12_layer_call_and_return_conditional_losses_3618730
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
:���������	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
/__inference_expand_11_5x5_layer_call_fn_3618544

inputs!
unknown:	 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_11_5x5_layer_call_and_return_conditional_losses_3614783w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
J__inference_expand_20_5x5_layer_call_and_return_conditional_losses_3615242

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:��������� m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
0__inference_contract_1_5x5_layer_call_fn_3617926

inputs"
unknown:�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_1_5x5_layer_call_and_return_conditional_losses_3614298w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
K__inference_contract_3_3x3_layer_call_and_return_conditional_losses_3614392

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
I__inference_expand_6_5x5_layer_call_and_return_conditional_losses_3618230

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:��������� m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
0__inference_contract_8_3x3_layer_call_fn_3618369

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_8_3x3_layer_call_and_return_conditional_losses_3614647w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
J__inference_expand_18_5x5_layer_call_and_return_conditional_losses_3619010

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:��������� m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
U
)__inference_skip_13_layer_call_fn_3618711
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_13_layer_call_and_return_conditional_losses_3614914h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
1__inference_contract_13_3x3_layer_call_fn_3618694

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_13_3x3_layer_call_and_return_conditional_losses_3614902w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�)
�
%__inference_signature_wrapper_3617857

inputs"
unknown:�
	unknown_0:	�$
	unknown_1:�
	unknown_2:	�$
	unknown_3:�
	unknown_4:$
	unknown_5:�
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

unknown_87:	�,

unknown_88:
identity

identity_1��StatefulPartitionedCall�
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
':����������:���������*|
_read_only_resource_inputs^
\Z	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ*0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__wrapped_model_3614229p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�*
�
/__inference_gomoku_resnet_layer_call_fn_3615544

inputs"
unknown:�
	unknown_0:	�$
	unknown_1:�
	unknown_2:	�$
	unknown_3:�
	unknown_4:$
	unknown_5:�
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

unknown_87:	�,

unknown_88:
identity

identity_1��StatefulPartitionedCall�
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
':����������:���������*|
_read_only_resource_inputs^
\Z	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gomoku_resnet_layer_call_and_return_conditional_losses_3615359p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
1__inference_contract_16_3x3_layer_call_fn_3618889

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_16_3x3_layer_call_and_return_conditional_losses_3615055w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
I__inference_expand_9_5x5_layer_call_and_return_conditional_losses_3614681

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:��������� m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
v
J__inference_concatenate_5_layer_call_and_return_conditional_losses_3618275
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
:���������	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
I__inference_expand_3_5x5_layer_call_and_return_conditional_losses_3618035

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:��������� m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
T
(__inference_skip_7_layer_call_fn_3618321
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_7_layer_call_and_return_conditional_losses_3614608h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
J__inference_expand_10_5x5_layer_call_and_return_conditional_losses_3614732

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:��������� m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
u
K__inference_concatenate_13_layer_call_and_return_conditional_losses_3614974

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
:���������	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
/__inference_expand_19_5x5_layer_call_fn_3619064

inputs!
unknown:	 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_19_5x5_layer_call_and_return_conditional_losses_3615191w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
J__inference_expand_17_5x5_layer_call_and_return_conditional_losses_3615089

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:��������� m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
.__inference_expand_4_5x5_layer_call_fn_3618089

inputs!
unknown:	 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_4_5x5_layer_call_and_return_conditional_losses_3614426w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
o
C__inference_skip_6_layer_call_and_return_conditional_losses_3618262
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
i
M__inference_flat_value_input_layer_call_and_return_conditional_losses_3619235

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����e  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������,Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������,"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
K__inference_contract_9_3x3_layer_call_and_return_conditional_losses_3614698

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
m
C__inference_skip_5_layer_call_and_return_conditional_losses_3614506

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
L__inference_contract_10_3x3_layer_call_and_return_conditional_losses_3618510

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
K__inference_contract_3_3x3_layer_call_and_return_conditional_losses_3618055

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
.__inference_expand_5_5x5_layer_call_fn_3618154

inputs!
unknown:	 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_5_5x5_layer_call_and_return_conditional_losses_3614477w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
J__inference_expand_15_5x5_layer_call_and_return_conditional_losses_3618815

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:��������� m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
L__inference_contract_19_3x3_layer_call_and_return_conditional_losses_3615208

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
r
H__inference_concatenate_layer_call_and_return_conditional_losses_3614311

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
:���������	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
N__inference_policy_aggregator_layer_call_and_return_conditional_losses_3619192

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
K__inference_contract_5_3x3_layer_call_and_return_conditional_losses_3618185

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
/__inference_expand_10_5x5_layer_call_fn_3618479

inputs!
unknown:	 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_10_5x5_layer_call_and_return_conditional_losses_3614732w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
L__inference_contract_20_3x3_layer_call_and_return_conditional_losses_3615259

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
p
D__inference_skip_11_layer_call_and_return_conditional_losses_3618587
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
p
D__inference_skip_15_layer_call_and_return_conditional_losses_3618847
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
w
K__inference_concatenate_13_layer_call_and_return_conditional_losses_3618795
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
:���������	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
1__inference_contract_19_3x3_layer_call_fn_3619084

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_19_3x3_layer_call_and_return_conditional_losses_3615208w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
J__inference_expand_14_5x5_layer_call_and_return_conditional_losses_3614936

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:��������� m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
I__inference_expand_8_5x5_layer_call_and_return_conditional_losses_3618360

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:��������� m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
,__inference_value_head_layer_call_fn_3619265

inputs
unknown:	�,
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_value_head_layer_call_and_return_conditional_losses_3615344o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������,: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������,
 
_user_specified_nameinputs
�
�
K__inference_contract_1_5x5_layer_call_and_return_conditional_losses_3617937

inputs9
conv2d_readvariableop_resource:�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
1__inference_contract_14_3x3_layer_call_fn_3618759

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_14_3x3_layer_call_and_return_conditional_losses_3614953w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
t
H__inference_concatenate_layer_call_and_return_conditional_losses_3617950
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
:���������	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
\
0__inference_concatenate_10_layer_call_fn_3618593
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_10_layer_call_and_return_conditional_losses_3614821h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
3__inference_policy_aggregator_layer_call_fn_3619181

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_policy_aggregator_layer_call_and_return_conditional_losses_3615293w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
L__inference_contract_12_3x3_layer_call_and_return_conditional_losses_3618640

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
Y
-__inference_concatenate_layer_call_fn_3617943
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_3614311h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
J__inference_expand_13_5x5_layer_call_and_return_conditional_losses_3618685

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:��������� m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
L__inference_contract_14_3x3_layer_call_and_return_conditional_losses_3614953

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
m
C__inference_skip_2_layer_call_and_return_conditional_losses_3614353

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
d
H__inference_flat_logits_layer_call_and_return_conditional_losses_3619246

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����i  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
T
(__inference_skip_5_layer_call_fn_3618191
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_5_layer_call_and_return_conditional_losses_3614506h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
L__inference_contract_15_3x3_layer_call_and_return_conditional_losses_3615004

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
T
(__inference_skip_3_layer_call_fn_3618061
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_3_layer_call_and_return_conditional_losses_3614404h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
p
D__inference_skip_20_layer_call_and_return_conditional_losses_3619172
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
U
)__inference_skip_11_layer_call_fn_3618581
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_11_layer_call_and_return_conditional_losses_3614812h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
Ё
�-
J__inference_gomoku_resnet_layer_call_and_return_conditional_losses_3615359

inputs1
expand_1_11x11_3614248:�%
expand_1_11x11_3614250:	�5
heuristic_detector_3614265:�)
heuristic_detector_3614267:	�5
heuristic_priority_3614282:�(
heuristic_priority_3614284:1
contract_1_5x5_3614299:�$
contract_1_5x5_3614301:.
expand_2_5x5_3614325:	 "
expand_2_5x5_3614327: 0
contract_2_3x3_3614342: $
contract_2_3x3_3614344:.
expand_3_5x5_3614376:	 "
expand_3_5x5_3614378: 0
contract_3_3x3_3614393: $
contract_3_3x3_3614395:.
expand_4_5x5_3614427:	 "
expand_4_5x5_3614429: 0
contract_4_3x3_3614444: $
contract_4_3x3_3614446:.
expand_5_5x5_3614478:	 "
expand_5_5x5_3614480: 0
contract_5_3x3_3614495: $
contract_5_3x3_3614497:.
expand_6_5x5_3614529:	 "
expand_6_5x5_3614531: 0
contract_6_3x3_3614546: $
contract_6_3x3_3614548:.
expand_7_5x5_3614580:	 "
expand_7_5x5_3614582: 0
contract_7_3x3_3614597: $
contract_7_3x3_3614599:.
expand_8_5x5_3614631:	 "
expand_8_5x5_3614633: 0
contract_8_3x3_3614648: $
contract_8_3x3_3614650:.
expand_9_5x5_3614682:	 "
expand_9_5x5_3614684: 0
contract_9_3x3_3614699: $
contract_9_3x3_3614701:/
expand_10_5x5_3614733:	 #
expand_10_5x5_3614735: 1
contract_10_3x3_3614750: %
contract_10_3x3_3614752:/
expand_11_5x5_3614784:	 #
expand_11_5x5_3614786: 1
contract_11_3x3_3614801: %
contract_11_3x3_3614803:/
expand_12_5x5_3614835:	 #
expand_12_5x5_3614837: 1
contract_12_3x3_3614852: %
contract_12_3x3_3614854:/
expand_13_5x5_3614886:	 #
expand_13_5x5_3614888: 1
contract_13_3x3_3614903: %
contract_13_3x3_3614905:/
expand_14_5x5_3614937:	 #
expand_14_5x5_3614939: 1
contract_14_3x3_3614954: %
contract_14_3x3_3614956:/
expand_15_5x5_3614988:	 #
expand_15_5x5_3614990: 1
contract_15_3x3_3615005: %
contract_15_3x3_3615007:/
expand_16_5x5_3615039:	 #
expand_16_5x5_3615041: 1
contract_16_3x3_3615056: %
contract_16_3x3_3615058:/
expand_17_5x5_3615090:	 #
expand_17_5x5_3615092: 1
contract_17_3x3_3615107: %
contract_17_3x3_3615109:/
expand_18_5x5_3615141:	 #
expand_18_5x5_3615143: 1
contract_18_3x3_3615158: %
contract_18_3x3_3615160:/
expand_19_5x5_3615192:	 #
expand_19_5x5_3615194: 1
contract_19_3x3_3615209: %
contract_19_3x3_3615211:/
expand_20_5x5_3615243:	 #
expand_20_5x5_3615245: 1
contract_20_3x3_3615260: %
contract_20_3x3_3615262:3
policy_aggregator_3615294:'
policy_aggregator_3615296:,
border_off_3615318: 
border_off_3615320:%
value_head_3615345:	�, 
value_head_3615347:
identity

identity_1��"border_off/StatefulPartitionedCall�'contract_10_3x3/StatefulPartitionedCall�'contract_11_3x3/StatefulPartitionedCall�'contract_12_3x3/StatefulPartitionedCall�'contract_13_3x3/StatefulPartitionedCall�'contract_14_3x3/StatefulPartitionedCall�'contract_15_3x3/StatefulPartitionedCall�'contract_16_3x3/StatefulPartitionedCall�'contract_17_3x3/StatefulPartitionedCall�'contract_18_3x3/StatefulPartitionedCall�'contract_19_3x3/StatefulPartitionedCall�&contract_1_5x5/StatefulPartitionedCall�'contract_20_3x3/StatefulPartitionedCall�&contract_2_3x3/StatefulPartitionedCall�&contract_3_3x3/StatefulPartitionedCall�&contract_4_3x3/StatefulPartitionedCall�&contract_5_3x3/StatefulPartitionedCall�&contract_6_3x3/StatefulPartitionedCall�&contract_7_3x3/StatefulPartitionedCall�&contract_8_3x3/StatefulPartitionedCall�&contract_9_3x3/StatefulPartitionedCall�%expand_10_5x5/StatefulPartitionedCall�%expand_11_5x5/StatefulPartitionedCall�%expand_12_5x5/StatefulPartitionedCall�%expand_13_5x5/StatefulPartitionedCall�%expand_14_5x5/StatefulPartitionedCall�%expand_15_5x5/StatefulPartitionedCall�%expand_16_5x5/StatefulPartitionedCall�%expand_17_5x5/StatefulPartitionedCall�%expand_18_5x5/StatefulPartitionedCall�%expand_19_5x5/StatefulPartitionedCall�&expand_1_11x11/StatefulPartitionedCall�%expand_20_5x5/StatefulPartitionedCall�$expand_2_5x5/StatefulPartitionedCall�$expand_3_5x5/StatefulPartitionedCall�$expand_4_5x5/StatefulPartitionedCall�$expand_5_5x5/StatefulPartitionedCall�$expand_6_5x5/StatefulPartitionedCall�$expand_7_5x5/StatefulPartitionedCall�$expand_8_5x5/StatefulPartitionedCall�$expand_9_5x5/StatefulPartitionedCall�*heuristic_detector/StatefulPartitionedCall�*heuristic_priority/StatefulPartitionedCall�)policy_aggregator/StatefulPartitionedCall�"value_head/StatefulPartitionedCall�
&expand_1_11x11/StatefulPartitionedCallStatefulPartitionedCallinputsexpand_1_11x11_3614248expand_1_11x11_3614250*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_expand_1_11x11_layer_call_and_return_conditional_losses_3614247�
*heuristic_detector/StatefulPartitionedCallStatefulPartitionedCallinputsheuristic_detector_3614265heuristic_detector_3614267*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_heuristic_detector_layer_call_and_return_conditional_losses_3614264�
*heuristic_priority/StatefulPartitionedCallStatefulPartitionedCall3heuristic_detector/StatefulPartitionedCall:output:0heuristic_priority_3614282heuristic_priority_3614284*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_heuristic_priority_layer_call_and_return_conditional_losses_3614281�
&contract_1_5x5/StatefulPartitionedCallStatefulPartitionedCall/expand_1_11x11/StatefulPartitionedCall:output:0contract_1_5x5_3614299contract_1_5x5_3614301*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_1_5x5_layer_call_and_return_conditional_losses_3614298�
concatenate/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0/contract_1_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_3614311�
$expand_2_5x5/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0expand_2_5x5_3614325expand_2_5x5_3614327*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_2_5x5_layer_call_and_return_conditional_losses_3614324�
&contract_2_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_2_5x5/StatefulPartitionedCall:output:0contract_2_3x3_3614342contract_2_3x3_3614344*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_2_3x3_layer_call_and_return_conditional_losses_3614341�
skip_2/PartitionedCallPartitionedCall/contract_2_3x3/StatefulPartitionedCall:output:0/contract_1_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_2_layer_call_and_return_conditional_losses_3614353�
concatenate_1/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_3614362�
$expand_3_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0expand_3_5x5_3614376expand_3_5x5_3614378*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_3_5x5_layer_call_and_return_conditional_losses_3614375�
&contract_3_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_3_5x5/StatefulPartitionedCall:output:0contract_3_3x3_3614393contract_3_3x3_3614395*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_3_3x3_layer_call_and_return_conditional_losses_3614392�
skip_3/PartitionedCallPartitionedCall/contract_3_3x3/StatefulPartitionedCall:output:0skip_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_3_layer_call_and_return_conditional_losses_3614404�
concatenate_2/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_3614413�
$expand_4_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0expand_4_5x5_3614427expand_4_5x5_3614429*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_4_5x5_layer_call_and_return_conditional_losses_3614426�
&contract_4_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_4_5x5/StatefulPartitionedCall:output:0contract_4_3x3_3614444contract_4_3x3_3614446*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_4_3x3_layer_call_and_return_conditional_losses_3614443�
skip_4/PartitionedCallPartitionedCall/contract_4_3x3/StatefulPartitionedCall:output:0skip_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_4_layer_call_and_return_conditional_losses_3614455�
concatenate_3/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_3_layer_call_and_return_conditional_losses_3614464�
$expand_5_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0expand_5_5x5_3614478expand_5_5x5_3614480*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_5_5x5_layer_call_and_return_conditional_losses_3614477�
&contract_5_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_5_5x5/StatefulPartitionedCall:output:0contract_5_3x3_3614495contract_5_3x3_3614497*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_5_3x3_layer_call_and_return_conditional_losses_3614494�
skip_5/PartitionedCallPartitionedCall/contract_5_3x3/StatefulPartitionedCall:output:0skip_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_5_layer_call_and_return_conditional_losses_3614506�
concatenate_4/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_4_layer_call_and_return_conditional_losses_3614515�
$expand_6_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0expand_6_5x5_3614529expand_6_5x5_3614531*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_6_5x5_layer_call_and_return_conditional_losses_3614528�
&contract_6_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_6_5x5/StatefulPartitionedCall:output:0contract_6_3x3_3614546contract_6_3x3_3614548*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_6_3x3_layer_call_and_return_conditional_losses_3614545�
skip_6/PartitionedCallPartitionedCall/contract_6_3x3/StatefulPartitionedCall:output:0skip_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_6_layer_call_and_return_conditional_losses_3614557�
concatenate_5/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_5_layer_call_and_return_conditional_losses_3614566�
$expand_7_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0expand_7_5x5_3614580expand_7_5x5_3614582*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_7_5x5_layer_call_and_return_conditional_losses_3614579�
&contract_7_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_7_5x5/StatefulPartitionedCall:output:0contract_7_3x3_3614597contract_7_3x3_3614599*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_7_3x3_layer_call_and_return_conditional_losses_3614596�
skip_7/PartitionedCallPartitionedCall/contract_7_3x3/StatefulPartitionedCall:output:0skip_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_7_layer_call_and_return_conditional_losses_3614608�
concatenate_6/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_6_layer_call_and_return_conditional_losses_3614617�
$expand_8_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_6/PartitionedCall:output:0expand_8_5x5_3614631expand_8_5x5_3614633*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_8_5x5_layer_call_and_return_conditional_losses_3614630�
&contract_8_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_8_5x5/StatefulPartitionedCall:output:0contract_8_3x3_3614648contract_8_3x3_3614650*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_8_3x3_layer_call_and_return_conditional_losses_3614647�
skip_8/PartitionedCallPartitionedCall/contract_8_3x3/StatefulPartitionedCall:output:0skip_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_8_layer_call_and_return_conditional_losses_3614659�
concatenate_7/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_7_layer_call_and_return_conditional_losses_3614668�
$expand_9_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_7/PartitionedCall:output:0expand_9_5x5_3614682expand_9_5x5_3614684*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_9_5x5_layer_call_and_return_conditional_losses_3614681�
&contract_9_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_9_5x5/StatefulPartitionedCall:output:0contract_9_3x3_3614699contract_9_3x3_3614701*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_9_3x3_layer_call_and_return_conditional_losses_3614698�
skip_9/PartitionedCallPartitionedCall/contract_9_3x3/StatefulPartitionedCall:output:0skip_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_9_layer_call_and_return_conditional_losses_3614710�
concatenate_8/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_8_layer_call_and_return_conditional_losses_3614719�
%expand_10_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_8/PartitionedCall:output:0expand_10_5x5_3614733expand_10_5x5_3614735*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_10_5x5_layer_call_and_return_conditional_losses_3614732�
'contract_10_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_10_5x5/StatefulPartitionedCall:output:0contract_10_3x3_3614750contract_10_3x3_3614752*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_10_3x3_layer_call_and_return_conditional_losses_3614749�
skip_10/PartitionedCallPartitionedCall0contract_10_3x3/StatefulPartitionedCall:output:0skip_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_10_layer_call_and_return_conditional_losses_3614761�
concatenate_9/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_9_layer_call_and_return_conditional_losses_3614770�
%expand_11_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_9/PartitionedCall:output:0expand_11_5x5_3614784expand_11_5x5_3614786*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_11_5x5_layer_call_and_return_conditional_losses_3614783�
'contract_11_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_11_5x5/StatefulPartitionedCall:output:0contract_11_3x3_3614801contract_11_3x3_3614803*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_11_3x3_layer_call_and_return_conditional_losses_3614800�
skip_11/PartitionedCallPartitionedCall0contract_11_3x3/StatefulPartitionedCall:output:0 skip_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_11_layer_call_and_return_conditional_losses_3614812�
concatenate_10/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_10_layer_call_and_return_conditional_losses_3614821�
%expand_12_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_10/PartitionedCall:output:0expand_12_5x5_3614835expand_12_5x5_3614837*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_12_5x5_layer_call_and_return_conditional_losses_3614834�
'contract_12_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_12_5x5/StatefulPartitionedCall:output:0contract_12_3x3_3614852contract_12_3x3_3614854*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_12_3x3_layer_call_and_return_conditional_losses_3614851�
skip_12/PartitionedCallPartitionedCall0contract_12_3x3/StatefulPartitionedCall:output:0 skip_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_12_layer_call_and_return_conditional_losses_3614863�
concatenate_11/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_11_layer_call_and_return_conditional_losses_3614872�
%expand_13_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_11/PartitionedCall:output:0expand_13_5x5_3614886expand_13_5x5_3614888*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_13_5x5_layer_call_and_return_conditional_losses_3614885�
'contract_13_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_13_5x5/StatefulPartitionedCall:output:0contract_13_3x3_3614903contract_13_3x3_3614905*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_13_3x3_layer_call_and_return_conditional_losses_3614902�
skip_13/PartitionedCallPartitionedCall0contract_13_3x3/StatefulPartitionedCall:output:0 skip_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_13_layer_call_and_return_conditional_losses_3614914�
concatenate_12/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_12_layer_call_and_return_conditional_losses_3614923�
%expand_14_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_12/PartitionedCall:output:0expand_14_5x5_3614937expand_14_5x5_3614939*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_14_5x5_layer_call_and_return_conditional_losses_3614936�
'contract_14_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_14_5x5/StatefulPartitionedCall:output:0contract_14_3x3_3614954contract_14_3x3_3614956*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_14_3x3_layer_call_and_return_conditional_losses_3614953�
skip_14/PartitionedCallPartitionedCall0contract_14_3x3/StatefulPartitionedCall:output:0 skip_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_14_layer_call_and_return_conditional_losses_3614965�
concatenate_13/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_14/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_13_layer_call_and_return_conditional_losses_3614974�
%expand_15_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_13/PartitionedCall:output:0expand_15_5x5_3614988expand_15_5x5_3614990*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_15_5x5_layer_call_and_return_conditional_losses_3614987�
'contract_15_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_15_5x5/StatefulPartitionedCall:output:0contract_15_3x3_3615005contract_15_3x3_3615007*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_15_3x3_layer_call_and_return_conditional_losses_3615004�
skip_15/PartitionedCallPartitionedCall0contract_15_3x3/StatefulPartitionedCall:output:0 skip_14/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_15_layer_call_and_return_conditional_losses_3615016�
concatenate_14/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_14_layer_call_and_return_conditional_losses_3615025�
%expand_16_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_14/PartitionedCall:output:0expand_16_5x5_3615039expand_16_5x5_3615041*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_16_5x5_layer_call_and_return_conditional_losses_3615038�
'contract_16_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_16_5x5/StatefulPartitionedCall:output:0contract_16_3x3_3615056contract_16_3x3_3615058*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_16_3x3_layer_call_and_return_conditional_losses_3615055�
skip_16/PartitionedCallPartitionedCall0contract_16_3x3/StatefulPartitionedCall:output:0 skip_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_16_layer_call_and_return_conditional_losses_3615067�
concatenate_15/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_15_layer_call_and_return_conditional_losses_3615076�
%expand_17_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_15/PartitionedCall:output:0expand_17_5x5_3615090expand_17_5x5_3615092*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_17_5x5_layer_call_and_return_conditional_losses_3615089�
'contract_17_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_17_5x5/StatefulPartitionedCall:output:0contract_17_3x3_3615107contract_17_3x3_3615109*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_17_3x3_layer_call_and_return_conditional_losses_3615106�
skip_17/PartitionedCallPartitionedCall0contract_17_3x3/StatefulPartitionedCall:output:0 skip_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_17_layer_call_and_return_conditional_losses_3615118�
concatenate_16/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_16_layer_call_and_return_conditional_losses_3615127�
%expand_18_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_16/PartitionedCall:output:0expand_18_5x5_3615141expand_18_5x5_3615143*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_18_5x5_layer_call_and_return_conditional_losses_3615140�
'contract_18_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_18_5x5/StatefulPartitionedCall:output:0contract_18_3x3_3615158contract_18_3x3_3615160*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_18_3x3_layer_call_and_return_conditional_losses_3615157�
skip_18/PartitionedCallPartitionedCall0contract_18_3x3/StatefulPartitionedCall:output:0 skip_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_18_layer_call_and_return_conditional_losses_3615169�
concatenate_17/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_18/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_17_layer_call_and_return_conditional_losses_3615178�
%expand_19_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_17/PartitionedCall:output:0expand_19_5x5_3615192expand_19_5x5_3615194*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_19_5x5_layer_call_and_return_conditional_losses_3615191�
'contract_19_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_19_5x5/StatefulPartitionedCall:output:0contract_19_3x3_3615209contract_19_3x3_3615211*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_19_3x3_layer_call_and_return_conditional_losses_3615208�
skip_19/PartitionedCallPartitionedCall0contract_19_3x3/StatefulPartitionedCall:output:0 skip_18/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_19_layer_call_and_return_conditional_losses_3615220�
concatenate_18/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_19/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_18_layer_call_and_return_conditional_losses_3615229�
%expand_20_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_18/PartitionedCall:output:0expand_20_5x5_3615243expand_20_5x5_3615245*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_20_5x5_layer_call_and_return_conditional_losses_3615242�
'contract_20_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_20_5x5/StatefulPartitionedCall:output:0contract_20_3x3_3615260contract_20_3x3_3615262*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_20_3x3_layer_call_and_return_conditional_losses_3615259�
skip_20/PartitionedCallPartitionedCall0contract_20_3x3/StatefulPartitionedCall:output:0 skip_19/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_20_layer_call_and_return_conditional_losses_3615271�
all_value_input/PartitionedCallPartitionedCall0contract_20_3x3/StatefulPartitionedCall:output:0$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_all_value_input_layer_call_and_return_conditional_losses_3615280�
)policy_aggregator/StatefulPartitionedCallStatefulPartitionedCall skip_20/PartitionedCall:output:0policy_aggregator_3615294policy_aggregator_3615296*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_policy_aggregator_layer_call_and_return_conditional_losses_3615293�
 flat_value_input/PartitionedCallPartitionedCall(all_value_input/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������,* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_flat_value_input_layer_call_and_return_conditional_losses_3615305�
"border_off/StatefulPartitionedCallStatefulPartitionedCall2policy_aggregator/StatefulPartitionedCall:output:0border_off_3615318border_off_3615320*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_border_off_layer_call_and_return_conditional_losses_3615317^
tf.math.truediv/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �B�
tf.math.truediv/truedivRealDiv)flat_value_input/PartitionedCall:output:0"tf.math.truediv/truediv/y:output:0*
T0*(
_output_shapes
:����������,�
flat_logits/PartitionedCallPartitionedCall+border_off/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_flat_logits_layer_call_and_return_conditional_losses_3615331�
"value_head/StatefulPartitionedCallStatefulPartitionedCalltf.math.truediv/truediv:z:0value_head_3615345value_head_3615347*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_value_head_layer_call_and_return_conditional_losses_3615344�
policy_head/PartitionedCallPartitionedCall$flat_logits/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_policy_head_layer_call_and_return_conditional_losses_3615355t
IdentityIdentity$policy_head/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������|

Identity_1Identity+value_head/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp#^border_off/StatefulPartitionedCall(^contract_10_3x3/StatefulPartitionedCall(^contract_11_3x3/StatefulPartitionedCall(^contract_12_3x3/StatefulPartitionedCall(^contract_13_3x3/StatefulPartitionedCall(^contract_14_3x3/StatefulPartitionedCall(^contract_15_3x3/StatefulPartitionedCall(^contract_16_3x3/StatefulPartitionedCall(^contract_17_3x3/StatefulPartitionedCall(^contract_18_3x3/StatefulPartitionedCall(^contract_19_3x3/StatefulPartitionedCall'^contract_1_5x5/StatefulPartitionedCall(^contract_20_3x3/StatefulPartitionedCall'^contract_2_3x3/StatefulPartitionedCall'^contract_3_3x3/StatefulPartitionedCall'^contract_4_3x3/StatefulPartitionedCall'^contract_5_3x3/StatefulPartitionedCall'^contract_6_3x3/StatefulPartitionedCall'^contract_7_3x3/StatefulPartitionedCall'^contract_8_3x3/StatefulPartitionedCall'^contract_9_3x3/StatefulPartitionedCall&^expand_10_5x5/StatefulPartitionedCall&^expand_11_5x5/StatefulPartitionedCall&^expand_12_5x5/StatefulPartitionedCall&^expand_13_5x5/StatefulPartitionedCall&^expand_14_5x5/StatefulPartitionedCall&^expand_15_5x5/StatefulPartitionedCall&^expand_16_5x5/StatefulPartitionedCall&^expand_17_5x5/StatefulPartitionedCall&^expand_18_5x5/StatefulPartitionedCall&^expand_19_5x5/StatefulPartitionedCall'^expand_1_11x11/StatefulPartitionedCall&^expand_20_5x5/StatefulPartitionedCall%^expand_2_5x5/StatefulPartitionedCall%^expand_3_5x5/StatefulPartitionedCall%^expand_4_5x5/StatefulPartitionedCall%^expand_5_5x5/StatefulPartitionedCall%^expand_6_5x5/StatefulPartitionedCall%^expand_7_5x5/StatefulPartitionedCall%^expand_8_5x5/StatefulPartitionedCall%^expand_9_5x5/StatefulPartitionedCall+^heuristic_detector/StatefulPartitionedCall+^heuristic_priority/StatefulPartitionedCall*^policy_aggregator/StatefulPartitionedCall#^value_head/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
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
:���������
 
_user_specified_nameinputs
�
�
I__inference_expand_7_5x5_layer_call_and_return_conditional_losses_3614579

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:��������� m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
I__inference_expand_9_5x5_layer_call_and_return_conditional_losses_3618425

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:��������� m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
/__inference_expand_16_5x5_layer_call_fn_3618869

inputs!
unknown:	 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_16_5x5_layer_call_and_return_conditional_losses_3615038w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
m
C__inference_skip_3_layer_call_and_return_conditional_losses_3614404

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
I__inference_expand_2_5x5_layer_call_and_return_conditional_losses_3614324

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:��������� m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
[
/__inference_concatenate_2_layer_call_fn_3618073
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_3614413h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
u
K__inference_concatenate_11_layer_call_and_return_conditional_losses_3614872

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
:���������	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
w
K__inference_concatenate_17_layer_call_and_return_conditional_losses_3619055
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
:���������	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
U
)__inference_skip_20_layer_call_fn_3619166
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_20_layer_call_and_return_conditional_losses_3615271h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
\
0__inference_concatenate_17_layer_call_fn_3619048
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_17_layer_call_and_return_conditional_losses_3615178h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
[
/__inference_concatenate_7_layer_call_fn_3618398
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_7_layer_call_and_return_conditional_losses_3614668h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
U
)__inference_skip_19_layer_call_fn_3619101
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_19_layer_call_and_return_conditional_losses_3615220h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
K__inference_contract_2_3x3_layer_call_and_return_conditional_losses_3617990

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
��
�<
#__inference__traced_restore_3619862
file_prefixE
*assignvariableop_heuristic_detector_kernel:�9
*assignvariableop_1_heuristic_detector_bias:	�C
(assignvariableop_2_expand_1_11x11_kernel:�5
&assignvariableop_3_expand_1_11x11_bias:	�G
,assignvariableop_4_heuristic_priority_kernel:�8
*assignvariableop_5_heuristic_priority_bias:C
(assignvariableop_6_contract_1_5x5_kernel:�4
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
%assignvariableop_88_value_head_kernel:	�,1
#assignvariableop_89_value_head_bias:#
assignvariableop_90_total: #
assignvariableop_91_count: 
identity_93��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_83�AssignVariableOp_84�AssignVariableOp_85�AssignVariableOp_86�AssignVariableOp_87�AssignVariableOp_88�AssignVariableOp_89�AssignVariableOp_9�AssignVariableOp_90�AssignVariableOp_91�)
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:]*
dtype0*�(
value�(B�(]B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-25/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-25/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-26/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-27/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-27/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-28/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-28/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-29/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-29/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-30/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-30/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-31/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-31/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-32/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-32/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-33/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-33/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-34/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-34/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-35/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-35/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-36/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-36/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-37/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-37/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-38/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-38/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-39/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-39/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-40/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-40/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-41/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-41/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-42/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-42/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-43/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-43/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-44/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-44/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:]*
dtype0*�
value�B�]B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*k
dtypesa
_2][
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp*assignvariableop_heuristic_detector_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp*assignvariableop_1_heuristic_detector_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp(assignvariableop_2_expand_1_11x11_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp&assignvariableop_3_expand_1_11x11_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp,assignvariableop_4_heuristic_priority_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp*assignvariableop_5_heuristic_priority_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp(assignvariableop_6_contract_1_5x5_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp&assignvariableop_7_contract_1_5x5_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp&assignvariableop_8_expand_2_5x5_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp$assignvariableop_9_expand_2_5x5_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp)assignvariableop_10_contract_2_3x3_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp'assignvariableop_11_contract_2_3x3_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp'assignvariableop_12_expand_3_5x5_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp%assignvariableop_13_expand_3_5x5_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp)assignvariableop_14_contract_3_3x3_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp'assignvariableop_15_contract_3_3x3_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp'assignvariableop_16_expand_4_5x5_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp%assignvariableop_17_expand_4_5x5_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp)assignvariableop_18_contract_4_3x3_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp'assignvariableop_19_contract_4_3x3_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp'assignvariableop_20_expand_5_5x5_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp%assignvariableop_21_expand_5_5x5_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp)assignvariableop_22_contract_5_3x3_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp'assignvariableop_23_contract_5_3x3_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp'assignvariableop_24_expand_6_5x5_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp%assignvariableop_25_expand_6_5x5_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp)assignvariableop_26_contract_6_3x3_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp'assignvariableop_27_contract_6_3x3_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp'assignvariableop_28_expand_7_5x5_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp%assignvariableop_29_expand_7_5x5_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp)assignvariableop_30_contract_7_3x3_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp'assignvariableop_31_contract_7_3x3_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp'assignvariableop_32_expand_8_5x5_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp%assignvariableop_33_expand_8_5x5_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp)assignvariableop_34_contract_8_3x3_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp'assignvariableop_35_contract_8_3x3_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp'assignvariableop_36_expand_9_5x5_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp%assignvariableop_37_expand_9_5x5_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp)assignvariableop_38_contract_9_3x3_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp'assignvariableop_39_contract_9_3x3_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp(assignvariableop_40_expand_10_5x5_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp&assignvariableop_41_expand_10_5x5_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp*assignvariableop_42_contract_10_3x3_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp(assignvariableop_43_contract_10_3x3_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp(assignvariableop_44_expand_11_5x5_kernelIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp&assignvariableop_45_expand_11_5x5_biasIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp*assignvariableop_46_contract_11_3x3_kernelIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp(assignvariableop_47_contract_11_3x3_biasIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp(assignvariableop_48_expand_12_5x5_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp&assignvariableop_49_expand_12_5x5_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp*assignvariableop_50_contract_12_3x3_kernelIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp(assignvariableop_51_contract_12_3x3_biasIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp(assignvariableop_52_expand_13_5x5_kernelIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp&assignvariableop_53_expand_13_5x5_biasIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp*assignvariableop_54_contract_13_3x3_kernelIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp(assignvariableop_55_contract_13_3x3_biasIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp(assignvariableop_56_expand_14_5x5_kernelIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp&assignvariableop_57_expand_14_5x5_biasIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp*assignvariableop_58_contract_14_3x3_kernelIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp(assignvariableop_59_contract_14_3x3_biasIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp(assignvariableop_60_expand_15_5x5_kernelIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp&assignvariableop_61_expand_15_5x5_biasIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp*assignvariableop_62_contract_15_3x3_kernelIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp(assignvariableop_63_contract_15_3x3_biasIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp(assignvariableop_64_expand_16_5x5_kernelIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp&assignvariableop_65_expand_16_5x5_biasIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp*assignvariableop_66_contract_16_3x3_kernelIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp(assignvariableop_67_contract_16_3x3_biasIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOp(assignvariableop_68_expand_17_5x5_kernelIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOp&assignvariableop_69_expand_17_5x5_biasIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOp*assignvariableop_70_contract_17_3x3_kernelIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOp(assignvariableop_71_contract_17_3x3_biasIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOp(assignvariableop_72_expand_18_5x5_kernelIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOp&assignvariableop_73_expand_18_5x5_biasIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_74AssignVariableOp*assignvariableop_74_contract_18_3x3_kernelIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOp(assignvariableop_75_contract_18_3x3_biasIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_76AssignVariableOp(assignvariableop_76_expand_19_5x5_kernelIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_77AssignVariableOp&assignvariableop_77_expand_19_5x5_biasIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_78AssignVariableOp*assignvariableop_78_contract_19_3x3_kernelIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_79AssignVariableOp(assignvariableop_79_contract_19_3x3_biasIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_80AssignVariableOp(assignvariableop_80_expand_20_5x5_kernelIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_81AssignVariableOp&assignvariableop_81_expand_20_5x5_biasIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_82AssignVariableOp*assignvariableop_82_contract_20_3x3_kernelIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_83AssignVariableOp(assignvariableop_83_contract_20_3x3_biasIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_84AssignVariableOp,assignvariableop_84_policy_aggregator_kernelIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_85AssignVariableOp*assignvariableop_85_policy_aggregator_biasIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_86AssignVariableOp%assignvariableop_86_border_off_kernelIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_87AssignVariableOp#assignvariableop_87_border_off_biasIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_88AssignVariableOp%assignvariableop_88_value_head_kernelIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_89AssignVariableOp#assignvariableop_89_value_head_biasIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_90AssignVariableOpassignvariableop_90_totalIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_91AssignVariableOpassignvariableop_91_countIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_92Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_93IdentityIdentity_92:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91*"
_acd_function_control_output(*
_output_shapes
 "#
identity_93Identity_93:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
�
t
J__inference_concatenate_7_layer_call_and_return_conditional_losses_3614668

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
:���������	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
o
C__inference_skip_5_layer_call_and_return_conditional_losses_3618197
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
L__inference_contract_12_3x3_layer_call_and_return_conditional_losses_3614851

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
o
C__inference_skip_9_layer_call_and_return_conditional_losses_3618457
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
p
D__inference_skip_17_layer_call_and_return_conditional_losses_3618977
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
/__inference_expand_15_5x5_layer_call_fn_3618804

inputs!
unknown:	 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_15_5x5_layer_call_and_return_conditional_losses_3614987w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
d
H__inference_policy_head_layer_call_and_return_conditional_losses_3619256

inputs
identityM
SoftmaxSoftmaxinputs*
T0*(
_output_shapes
:����������Z
IdentityIdentitySoftmax:softmax:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
/__inference_expand_20_5x5_layer_call_fn_3619129

inputs!
unknown:	 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_20_5x5_layer_call_and_return_conditional_losses_3615242w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
I__inference_expand_6_5x5_layer_call_and_return_conditional_losses_3614528

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:��������� m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
L__inference_contract_17_3x3_layer_call_and_return_conditional_losses_3615106

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
1__inference_contract_17_3x3_layer_call_fn_3618954

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_17_3x3_layer_call_and_return_conditional_losses_3615106w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
J__inference_expand_15_5x5_layer_call_and_return_conditional_losses_3614987

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:��������� m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
L__inference_contract_14_3x3_layer_call_and_return_conditional_losses_3618770

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
I__inference_expand_8_5x5_layer_call_and_return_conditional_losses_3614630

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:��������� m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
J__inference_expand_16_5x5_layer_call_and_return_conditional_losses_3615038

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:��������� m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
[
/__inference_concatenate_9_layer_call_fn_3618528
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_9_layer_call_and_return_conditional_losses_3614770h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
I__inference_expand_5_5x5_layer_call_and_return_conditional_losses_3618165

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:��������� m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
]
1__inference_all_value_input_layer_call_fn_3619198
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_all_value_input_layer_call_and_return_conditional_losses_3615280h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������	:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������	
"
_user_specified_name
inputs/1
�
x
L__inference_all_value_input_layer_call_and_return_conditional_losses_3619205
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
:���������_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������	:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������	
"
_user_specified_name
inputs/1
�
�
.__inference_expand_9_5x5_layer_call_fn_3618414

inputs!
unknown:	 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_9_5x5_layer_call_and_return_conditional_losses_3614681w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
U
)__inference_skip_16_layer_call_fn_3618906
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_16_layer_call_and_return_conditional_losses_3615067h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
[
/__inference_concatenate_6_layer_call_fn_3618333
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_6_layer_call_and_return_conditional_losses_3614617h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
J__inference_expand_12_5x5_layer_call_and_return_conditional_losses_3618620

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:��������� m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
w
K__inference_concatenate_15_layer_call_and_return_conditional_losses_3618925
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
:���������	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
��
�_
"__inference__wrapped_model_3614229

inputsV
;gomoku_resnet_expand_1_11x11_conv2d_readvariableop_resource:�K
<gomoku_resnet_expand_1_11x11_biasadd_readvariableop_resource:	�Z
?gomoku_resnet_heuristic_detector_conv2d_readvariableop_resource:�O
@gomoku_resnet_heuristic_detector_biasadd_readvariableop_resource:	�Z
?gomoku_resnet_heuristic_priority_conv2d_readvariableop_resource:�N
@gomoku_resnet_heuristic_priority_biasadd_readvariableop_resource:V
;gomoku_resnet_contract_1_5x5_conv2d_readvariableop_resource:�J
<gomoku_resnet_contract_1_5x5_biasadd_readvariableop_resource:S
9gomoku_resnet_expand_2_5x5_conv2d_readvariableop_resource:	 H
:gomoku_resnet_expand_2_5x5_biasadd_readvariableop_resource: U
;gomoku_resnet_contract_2_3x3_conv2d_readvariableop_resource: J
<gomoku_resnet_contract_2_3x3_biasadd_readvariableop_resource:S
9gomoku_resnet_expand_3_5x5_conv2d_readvariableop_resource:	 H
:gomoku_resnet_expand_3_5x5_biasadd_readvariableop_resource: U
;gomoku_resnet_contract_3_3x3_conv2d_readvariableop_resource: J
<gomoku_resnet_contract_3_3x3_biasadd_readvariableop_resource:S
9gomoku_resnet_expand_4_5x5_conv2d_readvariableop_resource:	 H
:gomoku_resnet_expand_4_5x5_biasadd_readvariableop_resource: U
;gomoku_resnet_contract_4_3x3_conv2d_readvariableop_resource: J
<gomoku_resnet_contract_4_3x3_biasadd_readvariableop_resource:S
9gomoku_resnet_expand_5_5x5_conv2d_readvariableop_resource:	 H
:gomoku_resnet_expand_5_5x5_biasadd_readvariableop_resource: U
;gomoku_resnet_contract_5_3x3_conv2d_readvariableop_resource: J
<gomoku_resnet_contract_5_3x3_biasadd_readvariableop_resource:S
9gomoku_resnet_expand_6_5x5_conv2d_readvariableop_resource:	 H
:gomoku_resnet_expand_6_5x5_biasadd_readvariableop_resource: U
;gomoku_resnet_contract_6_3x3_conv2d_readvariableop_resource: J
<gomoku_resnet_contract_6_3x3_biasadd_readvariableop_resource:S
9gomoku_resnet_expand_7_5x5_conv2d_readvariableop_resource:	 H
:gomoku_resnet_expand_7_5x5_biasadd_readvariableop_resource: U
;gomoku_resnet_contract_7_3x3_conv2d_readvariableop_resource: J
<gomoku_resnet_contract_7_3x3_biasadd_readvariableop_resource:S
9gomoku_resnet_expand_8_5x5_conv2d_readvariableop_resource:	 H
:gomoku_resnet_expand_8_5x5_biasadd_readvariableop_resource: U
;gomoku_resnet_contract_8_3x3_conv2d_readvariableop_resource: J
<gomoku_resnet_contract_8_3x3_biasadd_readvariableop_resource:S
9gomoku_resnet_expand_9_5x5_conv2d_readvariableop_resource:	 H
:gomoku_resnet_expand_9_5x5_biasadd_readvariableop_resource: U
;gomoku_resnet_contract_9_3x3_conv2d_readvariableop_resource: J
<gomoku_resnet_contract_9_3x3_biasadd_readvariableop_resource:T
:gomoku_resnet_expand_10_5x5_conv2d_readvariableop_resource:	 I
;gomoku_resnet_expand_10_5x5_biasadd_readvariableop_resource: V
<gomoku_resnet_contract_10_3x3_conv2d_readvariableop_resource: K
=gomoku_resnet_contract_10_3x3_biasadd_readvariableop_resource:T
:gomoku_resnet_expand_11_5x5_conv2d_readvariableop_resource:	 I
;gomoku_resnet_expand_11_5x5_biasadd_readvariableop_resource: V
<gomoku_resnet_contract_11_3x3_conv2d_readvariableop_resource: K
=gomoku_resnet_contract_11_3x3_biasadd_readvariableop_resource:T
:gomoku_resnet_expand_12_5x5_conv2d_readvariableop_resource:	 I
;gomoku_resnet_expand_12_5x5_biasadd_readvariableop_resource: V
<gomoku_resnet_contract_12_3x3_conv2d_readvariableop_resource: K
=gomoku_resnet_contract_12_3x3_biasadd_readvariableop_resource:T
:gomoku_resnet_expand_13_5x5_conv2d_readvariableop_resource:	 I
;gomoku_resnet_expand_13_5x5_biasadd_readvariableop_resource: V
<gomoku_resnet_contract_13_3x3_conv2d_readvariableop_resource: K
=gomoku_resnet_contract_13_3x3_biasadd_readvariableop_resource:T
:gomoku_resnet_expand_14_5x5_conv2d_readvariableop_resource:	 I
;gomoku_resnet_expand_14_5x5_biasadd_readvariableop_resource: V
<gomoku_resnet_contract_14_3x3_conv2d_readvariableop_resource: K
=gomoku_resnet_contract_14_3x3_biasadd_readvariableop_resource:T
:gomoku_resnet_expand_15_5x5_conv2d_readvariableop_resource:	 I
;gomoku_resnet_expand_15_5x5_biasadd_readvariableop_resource: V
<gomoku_resnet_contract_15_3x3_conv2d_readvariableop_resource: K
=gomoku_resnet_contract_15_3x3_biasadd_readvariableop_resource:T
:gomoku_resnet_expand_16_5x5_conv2d_readvariableop_resource:	 I
;gomoku_resnet_expand_16_5x5_biasadd_readvariableop_resource: V
<gomoku_resnet_contract_16_3x3_conv2d_readvariableop_resource: K
=gomoku_resnet_contract_16_3x3_biasadd_readvariableop_resource:T
:gomoku_resnet_expand_17_5x5_conv2d_readvariableop_resource:	 I
;gomoku_resnet_expand_17_5x5_biasadd_readvariableop_resource: V
<gomoku_resnet_contract_17_3x3_conv2d_readvariableop_resource: K
=gomoku_resnet_contract_17_3x3_biasadd_readvariableop_resource:T
:gomoku_resnet_expand_18_5x5_conv2d_readvariableop_resource:	 I
;gomoku_resnet_expand_18_5x5_biasadd_readvariableop_resource: V
<gomoku_resnet_contract_18_3x3_conv2d_readvariableop_resource: K
=gomoku_resnet_contract_18_3x3_biasadd_readvariableop_resource:T
:gomoku_resnet_expand_19_5x5_conv2d_readvariableop_resource:	 I
;gomoku_resnet_expand_19_5x5_biasadd_readvariableop_resource: V
<gomoku_resnet_contract_19_3x3_conv2d_readvariableop_resource: K
=gomoku_resnet_contract_19_3x3_biasadd_readvariableop_resource:T
:gomoku_resnet_expand_20_5x5_conv2d_readvariableop_resource:	 I
;gomoku_resnet_expand_20_5x5_biasadd_readvariableop_resource: V
<gomoku_resnet_contract_20_3x3_conv2d_readvariableop_resource: K
=gomoku_resnet_contract_20_3x3_biasadd_readvariableop_resource:X
>gomoku_resnet_policy_aggregator_conv2d_readvariableop_resource:M
?gomoku_resnet_policy_aggregator_biasadd_readvariableop_resource:Q
7gomoku_resnet_border_off_conv2d_readvariableop_resource:F
8gomoku_resnet_border_off_biasadd_readvariableop_resource:J
7gomoku_resnet_value_head_matmul_readvariableop_resource:	�,F
8gomoku_resnet_value_head_biasadd_readvariableop_resource:
identity

identity_1��/gomoku_resnet/border_off/BiasAdd/ReadVariableOp�.gomoku_resnet/border_off/Conv2D/ReadVariableOp�4gomoku_resnet/contract_10_3x3/BiasAdd/ReadVariableOp�3gomoku_resnet/contract_10_3x3/Conv2D/ReadVariableOp�4gomoku_resnet/contract_11_3x3/BiasAdd/ReadVariableOp�3gomoku_resnet/contract_11_3x3/Conv2D/ReadVariableOp�4gomoku_resnet/contract_12_3x3/BiasAdd/ReadVariableOp�3gomoku_resnet/contract_12_3x3/Conv2D/ReadVariableOp�4gomoku_resnet/contract_13_3x3/BiasAdd/ReadVariableOp�3gomoku_resnet/contract_13_3x3/Conv2D/ReadVariableOp�4gomoku_resnet/contract_14_3x3/BiasAdd/ReadVariableOp�3gomoku_resnet/contract_14_3x3/Conv2D/ReadVariableOp�4gomoku_resnet/contract_15_3x3/BiasAdd/ReadVariableOp�3gomoku_resnet/contract_15_3x3/Conv2D/ReadVariableOp�4gomoku_resnet/contract_16_3x3/BiasAdd/ReadVariableOp�3gomoku_resnet/contract_16_3x3/Conv2D/ReadVariableOp�4gomoku_resnet/contract_17_3x3/BiasAdd/ReadVariableOp�3gomoku_resnet/contract_17_3x3/Conv2D/ReadVariableOp�4gomoku_resnet/contract_18_3x3/BiasAdd/ReadVariableOp�3gomoku_resnet/contract_18_3x3/Conv2D/ReadVariableOp�4gomoku_resnet/contract_19_3x3/BiasAdd/ReadVariableOp�3gomoku_resnet/contract_19_3x3/Conv2D/ReadVariableOp�3gomoku_resnet/contract_1_5x5/BiasAdd/ReadVariableOp�2gomoku_resnet/contract_1_5x5/Conv2D/ReadVariableOp�4gomoku_resnet/contract_20_3x3/BiasAdd/ReadVariableOp�3gomoku_resnet/contract_20_3x3/Conv2D/ReadVariableOp�3gomoku_resnet/contract_2_3x3/BiasAdd/ReadVariableOp�2gomoku_resnet/contract_2_3x3/Conv2D/ReadVariableOp�3gomoku_resnet/contract_3_3x3/BiasAdd/ReadVariableOp�2gomoku_resnet/contract_3_3x3/Conv2D/ReadVariableOp�3gomoku_resnet/contract_4_3x3/BiasAdd/ReadVariableOp�2gomoku_resnet/contract_4_3x3/Conv2D/ReadVariableOp�3gomoku_resnet/contract_5_3x3/BiasAdd/ReadVariableOp�2gomoku_resnet/contract_5_3x3/Conv2D/ReadVariableOp�3gomoku_resnet/contract_6_3x3/BiasAdd/ReadVariableOp�2gomoku_resnet/contract_6_3x3/Conv2D/ReadVariableOp�3gomoku_resnet/contract_7_3x3/BiasAdd/ReadVariableOp�2gomoku_resnet/contract_7_3x3/Conv2D/ReadVariableOp�3gomoku_resnet/contract_8_3x3/BiasAdd/ReadVariableOp�2gomoku_resnet/contract_8_3x3/Conv2D/ReadVariableOp�3gomoku_resnet/contract_9_3x3/BiasAdd/ReadVariableOp�2gomoku_resnet/contract_9_3x3/Conv2D/ReadVariableOp�2gomoku_resnet/expand_10_5x5/BiasAdd/ReadVariableOp�1gomoku_resnet/expand_10_5x5/Conv2D/ReadVariableOp�2gomoku_resnet/expand_11_5x5/BiasAdd/ReadVariableOp�1gomoku_resnet/expand_11_5x5/Conv2D/ReadVariableOp�2gomoku_resnet/expand_12_5x5/BiasAdd/ReadVariableOp�1gomoku_resnet/expand_12_5x5/Conv2D/ReadVariableOp�2gomoku_resnet/expand_13_5x5/BiasAdd/ReadVariableOp�1gomoku_resnet/expand_13_5x5/Conv2D/ReadVariableOp�2gomoku_resnet/expand_14_5x5/BiasAdd/ReadVariableOp�1gomoku_resnet/expand_14_5x5/Conv2D/ReadVariableOp�2gomoku_resnet/expand_15_5x5/BiasAdd/ReadVariableOp�1gomoku_resnet/expand_15_5x5/Conv2D/ReadVariableOp�2gomoku_resnet/expand_16_5x5/BiasAdd/ReadVariableOp�1gomoku_resnet/expand_16_5x5/Conv2D/ReadVariableOp�2gomoku_resnet/expand_17_5x5/BiasAdd/ReadVariableOp�1gomoku_resnet/expand_17_5x5/Conv2D/ReadVariableOp�2gomoku_resnet/expand_18_5x5/BiasAdd/ReadVariableOp�1gomoku_resnet/expand_18_5x5/Conv2D/ReadVariableOp�2gomoku_resnet/expand_19_5x5/BiasAdd/ReadVariableOp�1gomoku_resnet/expand_19_5x5/Conv2D/ReadVariableOp�3gomoku_resnet/expand_1_11x11/BiasAdd/ReadVariableOp�2gomoku_resnet/expand_1_11x11/Conv2D/ReadVariableOp�2gomoku_resnet/expand_20_5x5/BiasAdd/ReadVariableOp�1gomoku_resnet/expand_20_5x5/Conv2D/ReadVariableOp�1gomoku_resnet/expand_2_5x5/BiasAdd/ReadVariableOp�0gomoku_resnet/expand_2_5x5/Conv2D/ReadVariableOp�1gomoku_resnet/expand_3_5x5/BiasAdd/ReadVariableOp�0gomoku_resnet/expand_3_5x5/Conv2D/ReadVariableOp�1gomoku_resnet/expand_4_5x5/BiasAdd/ReadVariableOp�0gomoku_resnet/expand_4_5x5/Conv2D/ReadVariableOp�1gomoku_resnet/expand_5_5x5/BiasAdd/ReadVariableOp�0gomoku_resnet/expand_5_5x5/Conv2D/ReadVariableOp�1gomoku_resnet/expand_6_5x5/BiasAdd/ReadVariableOp�0gomoku_resnet/expand_6_5x5/Conv2D/ReadVariableOp�1gomoku_resnet/expand_7_5x5/BiasAdd/ReadVariableOp�0gomoku_resnet/expand_7_5x5/Conv2D/ReadVariableOp�1gomoku_resnet/expand_8_5x5/BiasAdd/ReadVariableOp�0gomoku_resnet/expand_8_5x5/Conv2D/ReadVariableOp�1gomoku_resnet/expand_9_5x5/BiasAdd/ReadVariableOp�0gomoku_resnet/expand_9_5x5/Conv2D/ReadVariableOp�7gomoku_resnet/heuristic_detector/BiasAdd/ReadVariableOp�6gomoku_resnet/heuristic_detector/Conv2D/ReadVariableOp�7gomoku_resnet/heuristic_priority/BiasAdd/ReadVariableOp�6gomoku_resnet/heuristic_priority/Conv2D/ReadVariableOp�6gomoku_resnet/policy_aggregator/BiasAdd/ReadVariableOp�5gomoku_resnet/policy_aggregator/Conv2D/ReadVariableOp�/gomoku_resnet/value_head/BiasAdd/ReadVariableOp�.gomoku_resnet/value_head/MatMul/ReadVariableOp�
2gomoku_resnet/expand_1_11x11/Conv2D/ReadVariableOpReadVariableOp;gomoku_resnet_expand_1_11x11_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
#gomoku_resnet/expand_1_11x11/Conv2DConv2Dinputs:gomoku_resnet/expand_1_11x11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
3gomoku_resnet/expand_1_11x11/BiasAdd/ReadVariableOpReadVariableOp<gomoku_resnet_expand_1_11x11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$gomoku_resnet/expand_1_11x11/BiasAddBiasAdd,gomoku_resnet/expand_1_11x11/Conv2D:output:0;gomoku_resnet/expand_1_11x11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
%gomoku_resnet/expand_1_11x11/SoftplusSoftplus-gomoku_resnet/expand_1_11x11/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
6gomoku_resnet/heuristic_detector/Conv2D/ReadVariableOpReadVariableOp?gomoku_resnet_heuristic_detector_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
'gomoku_resnet/heuristic_detector/Conv2DConv2Dinputs>gomoku_resnet/heuristic_detector/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
7gomoku_resnet/heuristic_detector/BiasAdd/ReadVariableOpReadVariableOp@gomoku_resnet_heuristic_detector_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
(gomoku_resnet/heuristic_detector/BiasAddBiasAdd0gomoku_resnet/heuristic_detector/Conv2D:output:0?gomoku_resnet/heuristic_detector/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
%gomoku_resnet/heuristic_detector/ReluRelu1gomoku_resnet/heuristic_detector/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
6gomoku_resnet/heuristic_priority/Conv2D/ReadVariableOpReadVariableOp?gomoku_resnet_heuristic_priority_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
'gomoku_resnet/heuristic_priority/Conv2DConv2D3gomoku_resnet/heuristic_detector/Relu:activations:0>gomoku_resnet/heuristic_priority/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
7gomoku_resnet/heuristic_priority/BiasAdd/ReadVariableOpReadVariableOp@gomoku_resnet_heuristic_priority_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
(gomoku_resnet/heuristic_priority/BiasAddBiasAdd0gomoku_resnet/heuristic_priority/Conv2D:output:0?gomoku_resnet/heuristic_priority/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
%gomoku_resnet/heuristic_priority/TanhTanh1gomoku_resnet/heuristic_priority/BiasAdd:output:0*
T0*/
_output_shapes
:����������
2gomoku_resnet/contract_1_5x5/Conv2D/ReadVariableOpReadVariableOp;gomoku_resnet_contract_1_5x5_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
#gomoku_resnet/contract_1_5x5/Conv2DConv2D3gomoku_resnet/expand_1_11x11/Softplus:activations:0:gomoku_resnet/contract_1_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
3gomoku_resnet/contract_1_5x5/BiasAdd/ReadVariableOpReadVariableOp<gomoku_resnet_contract_1_5x5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
$gomoku_resnet/contract_1_5x5/BiasAddBiasAdd,gomoku_resnet/contract_1_5x5/Conv2D:output:0;gomoku_resnet/contract_1_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
%gomoku_resnet/contract_1_5x5/SoftplusSoftplus-gomoku_resnet/contract_1_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:���������g
%gomoku_resnet/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
 gomoku_resnet/concatenate/concatConcatV2)gomoku_resnet/heuristic_priority/Tanh:y:03gomoku_resnet/contract_1_5x5/Softplus:activations:0.gomoku_resnet/concatenate/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������	�
0gomoku_resnet/expand_2_5x5/Conv2D/ReadVariableOpReadVariableOp9gomoku_resnet_expand_2_5x5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
!gomoku_resnet/expand_2_5x5/Conv2DConv2D)gomoku_resnet/concatenate/concat:output:08gomoku_resnet/expand_2_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
1gomoku_resnet/expand_2_5x5/BiasAdd/ReadVariableOpReadVariableOp:gomoku_resnet_expand_2_5x5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
"gomoku_resnet/expand_2_5x5/BiasAddBiasAdd*gomoku_resnet/expand_2_5x5/Conv2D:output:09gomoku_resnet/expand_2_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
#gomoku_resnet/expand_2_5x5/SoftplusSoftplus+gomoku_resnet/expand_2_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
2gomoku_resnet/contract_2_3x3/Conv2D/ReadVariableOpReadVariableOp;gomoku_resnet_contract_2_3x3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
#gomoku_resnet/contract_2_3x3/Conv2DConv2D1gomoku_resnet/expand_2_5x5/Softplus:activations:0:gomoku_resnet/contract_2_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
3gomoku_resnet/contract_2_3x3/BiasAdd/ReadVariableOpReadVariableOp<gomoku_resnet_contract_2_3x3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
$gomoku_resnet/contract_2_3x3/BiasAddBiasAdd,gomoku_resnet/contract_2_3x3/Conv2D:output:0;gomoku_resnet/contract_2_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
%gomoku_resnet/contract_2_3x3/SoftplusSoftplus-gomoku_resnet/contract_2_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:����������
gomoku_resnet/skip_2/addAddV23gomoku_resnet/contract_2_3x3/Softplus:activations:03gomoku_resnet/contract_1_5x5/Softplus:activations:0*
T0*/
_output_shapes
:���������i
'gomoku_resnet/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
"gomoku_resnet/concatenate_1/concatConcatV2)gomoku_resnet/heuristic_priority/Tanh:y:0gomoku_resnet/skip_2/add:z:00gomoku_resnet/concatenate_1/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������	�
0gomoku_resnet/expand_3_5x5/Conv2D/ReadVariableOpReadVariableOp9gomoku_resnet_expand_3_5x5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
!gomoku_resnet/expand_3_5x5/Conv2DConv2D+gomoku_resnet/concatenate_1/concat:output:08gomoku_resnet/expand_3_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
1gomoku_resnet/expand_3_5x5/BiasAdd/ReadVariableOpReadVariableOp:gomoku_resnet_expand_3_5x5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
"gomoku_resnet/expand_3_5x5/BiasAddBiasAdd*gomoku_resnet/expand_3_5x5/Conv2D:output:09gomoku_resnet/expand_3_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
#gomoku_resnet/expand_3_5x5/SoftplusSoftplus+gomoku_resnet/expand_3_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
2gomoku_resnet/contract_3_3x3/Conv2D/ReadVariableOpReadVariableOp;gomoku_resnet_contract_3_3x3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
#gomoku_resnet/contract_3_3x3/Conv2DConv2D1gomoku_resnet/expand_3_5x5/Softplus:activations:0:gomoku_resnet/contract_3_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
3gomoku_resnet/contract_3_3x3/BiasAdd/ReadVariableOpReadVariableOp<gomoku_resnet_contract_3_3x3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
$gomoku_resnet/contract_3_3x3/BiasAddBiasAdd,gomoku_resnet/contract_3_3x3/Conv2D:output:0;gomoku_resnet/contract_3_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
%gomoku_resnet/contract_3_3x3/SoftplusSoftplus-gomoku_resnet/contract_3_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:����������
gomoku_resnet/skip_3/addAddV23gomoku_resnet/contract_3_3x3/Softplus:activations:0gomoku_resnet/skip_2/add:z:0*
T0*/
_output_shapes
:���������i
'gomoku_resnet/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
"gomoku_resnet/concatenate_2/concatConcatV2)gomoku_resnet/heuristic_priority/Tanh:y:0gomoku_resnet/skip_3/add:z:00gomoku_resnet/concatenate_2/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������	�
0gomoku_resnet/expand_4_5x5/Conv2D/ReadVariableOpReadVariableOp9gomoku_resnet_expand_4_5x5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
!gomoku_resnet/expand_4_5x5/Conv2DConv2D+gomoku_resnet/concatenate_2/concat:output:08gomoku_resnet/expand_4_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
1gomoku_resnet/expand_4_5x5/BiasAdd/ReadVariableOpReadVariableOp:gomoku_resnet_expand_4_5x5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
"gomoku_resnet/expand_4_5x5/BiasAddBiasAdd*gomoku_resnet/expand_4_5x5/Conv2D:output:09gomoku_resnet/expand_4_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
#gomoku_resnet/expand_4_5x5/SoftplusSoftplus+gomoku_resnet/expand_4_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
2gomoku_resnet/contract_4_3x3/Conv2D/ReadVariableOpReadVariableOp;gomoku_resnet_contract_4_3x3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
#gomoku_resnet/contract_4_3x3/Conv2DConv2D1gomoku_resnet/expand_4_5x5/Softplus:activations:0:gomoku_resnet/contract_4_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
3gomoku_resnet/contract_4_3x3/BiasAdd/ReadVariableOpReadVariableOp<gomoku_resnet_contract_4_3x3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
$gomoku_resnet/contract_4_3x3/BiasAddBiasAdd,gomoku_resnet/contract_4_3x3/Conv2D:output:0;gomoku_resnet/contract_4_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
%gomoku_resnet/contract_4_3x3/SoftplusSoftplus-gomoku_resnet/contract_4_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:����������
gomoku_resnet/skip_4/addAddV23gomoku_resnet/contract_4_3x3/Softplus:activations:0gomoku_resnet/skip_3/add:z:0*
T0*/
_output_shapes
:���������i
'gomoku_resnet/concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
"gomoku_resnet/concatenate_3/concatConcatV2)gomoku_resnet/heuristic_priority/Tanh:y:0gomoku_resnet/skip_4/add:z:00gomoku_resnet/concatenate_3/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������	�
0gomoku_resnet/expand_5_5x5/Conv2D/ReadVariableOpReadVariableOp9gomoku_resnet_expand_5_5x5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
!gomoku_resnet/expand_5_5x5/Conv2DConv2D+gomoku_resnet/concatenate_3/concat:output:08gomoku_resnet/expand_5_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
1gomoku_resnet/expand_5_5x5/BiasAdd/ReadVariableOpReadVariableOp:gomoku_resnet_expand_5_5x5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
"gomoku_resnet/expand_5_5x5/BiasAddBiasAdd*gomoku_resnet/expand_5_5x5/Conv2D:output:09gomoku_resnet/expand_5_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
#gomoku_resnet/expand_5_5x5/SoftplusSoftplus+gomoku_resnet/expand_5_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
2gomoku_resnet/contract_5_3x3/Conv2D/ReadVariableOpReadVariableOp;gomoku_resnet_contract_5_3x3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
#gomoku_resnet/contract_5_3x3/Conv2DConv2D1gomoku_resnet/expand_5_5x5/Softplus:activations:0:gomoku_resnet/contract_5_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
3gomoku_resnet/contract_5_3x3/BiasAdd/ReadVariableOpReadVariableOp<gomoku_resnet_contract_5_3x3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
$gomoku_resnet/contract_5_3x3/BiasAddBiasAdd,gomoku_resnet/contract_5_3x3/Conv2D:output:0;gomoku_resnet/contract_5_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
%gomoku_resnet/contract_5_3x3/SoftplusSoftplus-gomoku_resnet/contract_5_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:����������
gomoku_resnet/skip_5/addAddV23gomoku_resnet/contract_5_3x3/Softplus:activations:0gomoku_resnet/skip_4/add:z:0*
T0*/
_output_shapes
:���������i
'gomoku_resnet/concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
"gomoku_resnet/concatenate_4/concatConcatV2)gomoku_resnet/heuristic_priority/Tanh:y:0gomoku_resnet/skip_5/add:z:00gomoku_resnet/concatenate_4/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������	�
0gomoku_resnet/expand_6_5x5/Conv2D/ReadVariableOpReadVariableOp9gomoku_resnet_expand_6_5x5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
!gomoku_resnet/expand_6_5x5/Conv2DConv2D+gomoku_resnet/concatenate_4/concat:output:08gomoku_resnet/expand_6_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
1gomoku_resnet/expand_6_5x5/BiasAdd/ReadVariableOpReadVariableOp:gomoku_resnet_expand_6_5x5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
"gomoku_resnet/expand_6_5x5/BiasAddBiasAdd*gomoku_resnet/expand_6_5x5/Conv2D:output:09gomoku_resnet/expand_6_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
#gomoku_resnet/expand_6_5x5/SoftplusSoftplus+gomoku_resnet/expand_6_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
2gomoku_resnet/contract_6_3x3/Conv2D/ReadVariableOpReadVariableOp;gomoku_resnet_contract_6_3x3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
#gomoku_resnet/contract_6_3x3/Conv2DConv2D1gomoku_resnet/expand_6_5x5/Softplus:activations:0:gomoku_resnet/contract_6_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
3gomoku_resnet/contract_6_3x3/BiasAdd/ReadVariableOpReadVariableOp<gomoku_resnet_contract_6_3x3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
$gomoku_resnet/contract_6_3x3/BiasAddBiasAdd,gomoku_resnet/contract_6_3x3/Conv2D:output:0;gomoku_resnet/contract_6_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
%gomoku_resnet/contract_6_3x3/SoftplusSoftplus-gomoku_resnet/contract_6_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:����������
gomoku_resnet/skip_6/addAddV23gomoku_resnet/contract_6_3x3/Softplus:activations:0gomoku_resnet/skip_5/add:z:0*
T0*/
_output_shapes
:���������i
'gomoku_resnet/concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
"gomoku_resnet/concatenate_5/concatConcatV2)gomoku_resnet/heuristic_priority/Tanh:y:0gomoku_resnet/skip_6/add:z:00gomoku_resnet/concatenate_5/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������	�
0gomoku_resnet/expand_7_5x5/Conv2D/ReadVariableOpReadVariableOp9gomoku_resnet_expand_7_5x5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
!gomoku_resnet/expand_7_5x5/Conv2DConv2D+gomoku_resnet/concatenate_5/concat:output:08gomoku_resnet/expand_7_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
1gomoku_resnet/expand_7_5x5/BiasAdd/ReadVariableOpReadVariableOp:gomoku_resnet_expand_7_5x5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
"gomoku_resnet/expand_7_5x5/BiasAddBiasAdd*gomoku_resnet/expand_7_5x5/Conv2D:output:09gomoku_resnet/expand_7_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
#gomoku_resnet/expand_7_5x5/SoftplusSoftplus+gomoku_resnet/expand_7_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
2gomoku_resnet/contract_7_3x3/Conv2D/ReadVariableOpReadVariableOp;gomoku_resnet_contract_7_3x3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
#gomoku_resnet/contract_7_3x3/Conv2DConv2D1gomoku_resnet/expand_7_5x5/Softplus:activations:0:gomoku_resnet/contract_7_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
3gomoku_resnet/contract_7_3x3/BiasAdd/ReadVariableOpReadVariableOp<gomoku_resnet_contract_7_3x3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
$gomoku_resnet/contract_7_3x3/BiasAddBiasAdd,gomoku_resnet/contract_7_3x3/Conv2D:output:0;gomoku_resnet/contract_7_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
%gomoku_resnet/contract_7_3x3/SoftplusSoftplus-gomoku_resnet/contract_7_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:����������
gomoku_resnet/skip_7/addAddV23gomoku_resnet/contract_7_3x3/Softplus:activations:0gomoku_resnet/skip_6/add:z:0*
T0*/
_output_shapes
:���������i
'gomoku_resnet/concatenate_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
"gomoku_resnet/concatenate_6/concatConcatV2)gomoku_resnet/heuristic_priority/Tanh:y:0gomoku_resnet/skip_7/add:z:00gomoku_resnet/concatenate_6/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������	�
0gomoku_resnet/expand_8_5x5/Conv2D/ReadVariableOpReadVariableOp9gomoku_resnet_expand_8_5x5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
!gomoku_resnet/expand_8_5x5/Conv2DConv2D+gomoku_resnet/concatenate_6/concat:output:08gomoku_resnet/expand_8_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
1gomoku_resnet/expand_8_5x5/BiasAdd/ReadVariableOpReadVariableOp:gomoku_resnet_expand_8_5x5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
"gomoku_resnet/expand_8_5x5/BiasAddBiasAdd*gomoku_resnet/expand_8_5x5/Conv2D:output:09gomoku_resnet/expand_8_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
#gomoku_resnet/expand_8_5x5/SoftplusSoftplus+gomoku_resnet/expand_8_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
2gomoku_resnet/contract_8_3x3/Conv2D/ReadVariableOpReadVariableOp;gomoku_resnet_contract_8_3x3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
#gomoku_resnet/contract_8_3x3/Conv2DConv2D1gomoku_resnet/expand_8_5x5/Softplus:activations:0:gomoku_resnet/contract_8_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
3gomoku_resnet/contract_8_3x3/BiasAdd/ReadVariableOpReadVariableOp<gomoku_resnet_contract_8_3x3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
$gomoku_resnet/contract_8_3x3/BiasAddBiasAdd,gomoku_resnet/contract_8_3x3/Conv2D:output:0;gomoku_resnet/contract_8_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
%gomoku_resnet/contract_8_3x3/SoftplusSoftplus-gomoku_resnet/contract_8_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:����������
gomoku_resnet/skip_8/addAddV23gomoku_resnet/contract_8_3x3/Softplus:activations:0gomoku_resnet/skip_7/add:z:0*
T0*/
_output_shapes
:���������i
'gomoku_resnet/concatenate_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
"gomoku_resnet/concatenate_7/concatConcatV2)gomoku_resnet/heuristic_priority/Tanh:y:0gomoku_resnet/skip_8/add:z:00gomoku_resnet/concatenate_7/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������	�
0gomoku_resnet/expand_9_5x5/Conv2D/ReadVariableOpReadVariableOp9gomoku_resnet_expand_9_5x5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
!gomoku_resnet/expand_9_5x5/Conv2DConv2D+gomoku_resnet/concatenate_7/concat:output:08gomoku_resnet/expand_9_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
1gomoku_resnet/expand_9_5x5/BiasAdd/ReadVariableOpReadVariableOp:gomoku_resnet_expand_9_5x5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
"gomoku_resnet/expand_9_5x5/BiasAddBiasAdd*gomoku_resnet/expand_9_5x5/Conv2D:output:09gomoku_resnet/expand_9_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
#gomoku_resnet/expand_9_5x5/SoftplusSoftplus+gomoku_resnet/expand_9_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
2gomoku_resnet/contract_9_3x3/Conv2D/ReadVariableOpReadVariableOp;gomoku_resnet_contract_9_3x3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
#gomoku_resnet/contract_9_3x3/Conv2DConv2D1gomoku_resnet/expand_9_5x5/Softplus:activations:0:gomoku_resnet/contract_9_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
3gomoku_resnet/contract_9_3x3/BiasAdd/ReadVariableOpReadVariableOp<gomoku_resnet_contract_9_3x3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
$gomoku_resnet/contract_9_3x3/BiasAddBiasAdd,gomoku_resnet/contract_9_3x3/Conv2D:output:0;gomoku_resnet/contract_9_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
%gomoku_resnet/contract_9_3x3/SoftplusSoftplus-gomoku_resnet/contract_9_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:����������
gomoku_resnet/skip_9/addAddV23gomoku_resnet/contract_9_3x3/Softplus:activations:0gomoku_resnet/skip_8/add:z:0*
T0*/
_output_shapes
:���������i
'gomoku_resnet/concatenate_8/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
"gomoku_resnet/concatenate_8/concatConcatV2)gomoku_resnet/heuristic_priority/Tanh:y:0gomoku_resnet/skip_9/add:z:00gomoku_resnet/concatenate_8/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������	�
1gomoku_resnet/expand_10_5x5/Conv2D/ReadVariableOpReadVariableOp:gomoku_resnet_expand_10_5x5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
"gomoku_resnet/expand_10_5x5/Conv2DConv2D+gomoku_resnet/concatenate_8/concat:output:09gomoku_resnet/expand_10_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
2gomoku_resnet/expand_10_5x5/BiasAdd/ReadVariableOpReadVariableOp;gomoku_resnet_expand_10_5x5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
#gomoku_resnet/expand_10_5x5/BiasAddBiasAdd+gomoku_resnet/expand_10_5x5/Conv2D:output:0:gomoku_resnet/expand_10_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
$gomoku_resnet/expand_10_5x5/SoftplusSoftplus,gomoku_resnet/expand_10_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
3gomoku_resnet/contract_10_3x3/Conv2D/ReadVariableOpReadVariableOp<gomoku_resnet_contract_10_3x3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
$gomoku_resnet/contract_10_3x3/Conv2DConv2D2gomoku_resnet/expand_10_5x5/Softplus:activations:0;gomoku_resnet/contract_10_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
4gomoku_resnet/contract_10_3x3/BiasAdd/ReadVariableOpReadVariableOp=gomoku_resnet_contract_10_3x3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
%gomoku_resnet/contract_10_3x3/BiasAddBiasAdd-gomoku_resnet/contract_10_3x3/Conv2D:output:0<gomoku_resnet/contract_10_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
&gomoku_resnet/contract_10_3x3/SoftplusSoftplus.gomoku_resnet/contract_10_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:����������
gomoku_resnet/skip_10/addAddV24gomoku_resnet/contract_10_3x3/Softplus:activations:0gomoku_resnet/skip_9/add:z:0*
T0*/
_output_shapes
:���������i
'gomoku_resnet/concatenate_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
"gomoku_resnet/concatenate_9/concatConcatV2)gomoku_resnet/heuristic_priority/Tanh:y:0gomoku_resnet/skip_10/add:z:00gomoku_resnet/concatenate_9/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������	�
1gomoku_resnet/expand_11_5x5/Conv2D/ReadVariableOpReadVariableOp:gomoku_resnet_expand_11_5x5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
"gomoku_resnet/expand_11_5x5/Conv2DConv2D+gomoku_resnet/concatenate_9/concat:output:09gomoku_resnet/expand_11_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
2gomoku_resnet/expand_11_5x5/BiasAdd/ReadVariableOpReadVariableOp;gomoku_resnet_expand_11_5x5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
#gomoku_resnet/expand_11_5x5/BiasAddBiasAdd+gomoku_resnet/expand_11_5x5/Conv2D:output:0:gomoku_resnet/expand_11_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
$gomoku_resnet/expand_11_5x5/SoftplusSoftplus,gomoku_resnet/expand_11_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
3gomoku_resnet/contract_11_3x3/Conv2D/ReadVariableOpReadVariableOp<gomoku_resnet_contract_11_3x3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
$gomoku_resnet/contract_11_3x3/Conv2DConv2D2gomoku_resnet/expand_11_5x5/Softplus:activations:0;gomoku_resnet/contract_11_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
4gomoku_resnet/contract_11_3x3/BiasAdd/ReadVariableOpReadVariableOp=gomoku_resnet_contract_11_3x3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
%gomoku_resnet/contract_11_3x3/BiasAddBiasAdd-gomoku_resnet/contract_11_3x3/Conv2D:output:0<gomoku_resnet/contract_11_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
&gomoku_resnet/contract_11_3x3/SoftplusSoftplus.gomoku_resnet/contract_11_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:����������
gomoku_resnet/skip_11/addAddV24gomoku_resnet/contract_11_3x3/Softplus:activations:0gomoku_resnet/skip_10/add:z:0*
T0*/
_output_shapes
:���������j
(gomoku_resnet/concatenate_10/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
#gomoku_resnet/concatenate_10/concatConcatV2)gomoku_resnet/heuristic_priority/Tanh:y:0gomoku_resnet/skip_11/add:z:01gomoku_resnet/concatenate_10/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������	�
1gomoku_resnet/expand_12_5x5/Conv2D/ReadVariableOpReadVariableOp:gomoku_resnet_expand_12_5x5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
"gomoku_resnet/expand_12_5x5/Conv2DConv2D,gomoku_resnet/concatenate_10/concat:output:09gomoku_resnet/expand_12_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
2gomoku_resnet/expand_12_5x5/BiasAdd/ReadVariableOpReadVariableOp;gomoku_resnet_expand_12_5x5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
#gomoku_resnet/expand_12_5x5/BiasAddBiasAdd+gomoku_resnet/expand_12_5x5/Conv2D:output:0:gomoku_resnet/expand_12_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
$gomoku_resnet/expand_12_5x5/SoftplusSoftplus,gomoku_resnet/expand_12_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
3gomoku_resnet/contract_12_3x3/Conv2D/ReadVariableOpReadVariableOp<gomoku_resnet_contract_12_3x3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
$gomoku_resnet/contract_12_3x3/Conv2DConv2D2gomoku_resnet/expand_12_5x5/Softplus:activations:0;gomoku_resnet/contract_12_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
4gomoku_resnet/contract_12_3x3/BiasAdd/ReadVariableOpReadVariableOp=gomoku_resnet_contract_12_3x3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
%gomoku_resnet/contract_12_3x3/BiasAddBiasAdd-gomoku_resnet/contract_12_3x3/Conv2D:output:0<gomoku_resnet/contract_12_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
&gomoku_resnet/contract_12_3x3/SoftplusSoftplus.gomoku_resnet/contract_12_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:����������
gomoku_resnet/skip_12/addAddV24gomoku_resnet/contract_12_3x3/Softplus:activations:0gomoku_resnet/skip_11/add:z:0*
T0*/
_output_shapes
:���������j
(gomoku_resnet/concatenate_11/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
#gomoku_resnet/concatenate_11/concatConcatV2)gomoku_resnet/heuristic_priority/Tanh:y:0gomoku_resnet/skip_12/add:z:01gomoku_resnet/concatenate_11/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������	�
1gomoku_resnet/expand_13_5x5/Conv2D/ReadVariableOpReadVariableOp:gomoku_resnet_expand_13_5x5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
"gomoku_resnet/expand_13_5x5/Conv2DConv2D,gomoku_resnet/concatenate_11/concat:output:09gomoku_resnet/expand_13_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
2gomoku_resnet/expand_13_5x5/BiasAdd/ReadVariableOpReadVariableOp;gomoku_resnet_expand_13_5x5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
#gomoku_resnet/expand_13_5x5/BiasAddBiasAdd+gomoku_resnet/expand_13_5x5/Conv2D:output:0:gomoku_resnet/expand_13_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
$gomoku_resnet/expand_13_5x5/SoftplusSoftplus,gomoku_resnet/expand_13_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
3gomoku_resnet/contract_13_3x3/Conv2D/ReadVariableOpReadVariableOp<gomoku_resnet_contract_13_3x3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
$gomoku_resnet/contract_13_3x3/Conv2DConv2D2gomoku_resnet/expand_13_5x5/Softplus:activations:0;gomoku_resnet/contract_13_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
4gomoku_resnet/contract_13_3x3/BiasAdd/ReadVariableOpReadVariableOp=gomoku_resnet_contract_13_3x3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
%gomoku_resnet/contract_13_3x3/BiasAddBiasAdd-gomoku_resnet/contract_13_3x3/Conv2D:output:0<gomoku_resnet/contract_13_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
&gomoku_resnet/contract_13_3x3/SoftplusSoftplus.gomoku_resnet/contract_13_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:����������
gomoku_resnet/skip_13/addAddV24gomoku_resnet/contract_13_3x3/Softplus:activations:0gomoku_resnet/skip_12/add:z:0*
T0*/
_output_shapes
:���������j
(gomoku_resnet/concatenate_12/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
#gomoku_resnet/concatenate_12/concatConcatV2)gomoku_resnet/heuristic_priority/Tanh:y:0gomoku_resnet/skip_13/add:z:01gomoku_resnet/concatenate_12/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������	�
1gomoku_resnet/expand_14_5x5/Conv2D/ReadVariableOpReadVariableOp:gomoku_resnet_expand_14_5x5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
"gomoku_resnet/expand_14_5x5/Conv2DConv2D,gomoku_resnet/concatenate_12/concat:output:09gomoku_resnet/expand_14_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
2gomoku_resnet/expand_14_5x5/BiasAdd/ReadVariableOpReadVariableOp;gomoku_resnet_expand_14_5x5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
#gomoku_resnet/expand_14_5x5/BiasAddBiasAdd+gomoku_resnet/expand_14_5x5/Conv2D:output:0:gomoku_resnet/expand_14_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
$gomoku_resnet/expand_14_5x5/SoftplusSoftplus,gomoku_resnet/expand_14_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
3gomoku_resnet/contract_14_3x3/Conv2D/ReadVariableOpReadVariableOp<gomoku_resnet_contract_14_3x3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
$gomoku_resnet/contract_14_3x3/Conv2DConv2D2gomoku_resnet/expand_14_5x5/Softplus:activations:0;gomoku_resnet/contract_14_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
4gomoku_resnet/contract_14_3x3/BiasAdd/ReadVariableOpReadVariableOp=gomoku_resnet_contract_14_3x3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
%gomoku_resnet/contract_14_3x3/BiasAddBiasAdd-gomoku_resnet/contract_14_3x3/Conv2D:output:0<gomoku_resnet/contract_14_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
&gomoku_resnet/contract_14_3x3/SoftplusSoftplus.gomoku_resnet/contract_14_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:����������
gomoku_resnet/skip_14/addAddV24gomoku_resnet/contract_14_3x3/Softplus:activations:0gomoku_resnet/skip_13/add:z:0*
T0*/
_output_shapes
:���������j
(gomoku_resnet/concatenate_13/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
#gomoku_resnet/concatenate_13/concatConcatV2)gomoku_resnet/heuristic_priority/Tanh:y:0gomoku_resnet/skip_14/add:z:01gomoku_resnet/concatenate_13/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������	�
1gomoku_resnet/expand_15_5x5/Conv2D/ReadVariableOpReadVariableOp:gomoku_resnet_expand_15_5x5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
"gomoku_resnet/expand_15_5x5/Conv2DConv2D,gomoku_resnet/concatenate_13/concat:output:09gomoku_resnet/expand_15_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
2gomoku_resnet/expand_15_5x5/BiasAdd/ReadVariableOpReadVariableOp;gomoku_resnet_expand_15_5x5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
#gomoku_resnet/expand_15_5x5/BiasAddBiasAdd+gomoku_resnet/expand_15_5x5/Conv2D:output:0:gomoku_resnet/expand_15_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
$gomoku_resnet/expand_15_5x5/SoftplusSoftplus,gomoku_resnet/expand_15_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
3gomoku_resnet/contract_15_3x3/Conv2D/ReadVariableOpReadVariableOp<gomoku_resnet_contract_15_3x3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
$gomoku_resnet/contract_15_3x3/Conv2DConv2D2gomoku_resnet/expand_15_5x5/Softplus:activations:0;gomoku_resnet/contract_15_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
4gomoku_resnet/contract_15_3x3/BiasAdd/ReadVariableOpReadVariableOp=gomoku_resnet_contract_15_3x3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
%gomoku_resnet/contract_15_3x3/BiasAddBiasAdd-gomoku_resnet/contract_15_3x3/Conv2D:output:0<gomoku_resnet/contract_15_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
&gomoku_resnet/contract_15_3x3/SoftplusSoftplus.gomoku_resnet/contract_15_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:����������
gomoku_resnet/skip_15/addAddV24gomoku_resnet/contract_15_3x3/Softplus:activations:0gomoku_resnet/skip_14/add:z:0*
T0*/
_output_shapes
:���������j
(gomoku_resnet/concatenate_14/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
#gomoku_resnet/concatenate_14/concatConcatV2)gomoku_resnet/heuristic_priority/Tanh:y:0gomoku_resnet/skip_15/add:z:01gomoku_resnet/concatenate_14/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������	�
1gomoku_resnet/expand_16_5x5/Conv2D/ReadVariableOpReadVariableOp:gomoku_resnet_expand_16_5x5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
"gomoku_resnet/expand_16_5x5/Conv2DConv2D,gomoku_resnet/concatenate_14/concat:output:09gomoku_resnet/expand_16_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
2gomoku_resnet/expand_16_5x5/BiasAdd/ReadVariableOpReadVariableOp;gomoku_resnet_expand_16_5x5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
#gomoku_resnet/expand_16_5x5/BiasAddBiasAdd+gomoku_resnet/expand_16_5x5/Conv2D:output:0:gomoku_resnet/expand_16_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
$gomoku_resnet/expand_16_5x5/SoftplusSoftplus,gomoku_resnet/expand_16_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
3gomoku_resnet/contract_16_3x3/Conv2D/ReadVariableOpReadVariableOp<gomoku_resnet_contract_16_3x3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
$gomoku_resnet/contract_16_3x3/Conv2DConv2D2gomoku_resnet/expand_16_5x5/Softplus:activations:0;gomoku_resnet/contract_16_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
4gomoku_resnet/contract_16_3x3/BiasAdd/ReadVariableOpReadVariableOp=gomoku_resnet_contract_16_3x3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
%gomoku_resnet/contract_16_3x3/BiasAddBiasAdd-gomoku_resnet/contract_16_3x3/Conv2D:output:0<gomoku_resnet/contract_16_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
&gomoku_resnet/contract_16_3x3/SoftplusSoftplus.gomoku_resnet/contract_16_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:����������
gomoku_resnet/skip_16/addAddV24gomoku_resnet/contract_16_3x3/Softplus:activations:0gomoku_resnet/skip_15/add:z:0*
T0*/
_output_shapes
:���������j
(gomoku_resnet/concatenate_15/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
#gomoku_resnet/concatenate_15/concatConcatV2)gomoku_resnet/heuristic_priority/Tanh:y:0gomoku_resnet/skip_16/add:z:01gomoku_resnet/concatenate_15/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������	�
1gomoku_resnet/expand_17_5x5/Conv2D/ReadVariableOpReadVariableOp:gomoku_resnet_expand_17_5x5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
"gomoku_resnet/expand_17_5x5/Conv2DConv2D,gomoku_resnet/concatenate_15/concat:output:09gomoku_resnet/expand_17_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
2gomoku_resnet/expand_17_5x5/BiasAdd/ReadVariableOpReadVariableOp;gomoku_resnet_expand_17_5x5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
#gomoku_resnet/expand_17_5x5/BiasAddBiasAdd+gomoku_resnet/expand_17_5x5/Conv2D:output:0:gomoku_resnet/expand_17_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
$gomoku_resnet/expand_17_5x5/SoftplusSoftplus,gomoku_resnet/expand_17_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
3gomoku_resnet/contract_17_3x3/Conv2D/ReadVariableOpReadVariableOp<gomoku_resnet_contract_17_3x3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
$gomoku_resnet/contract_17_3x3/Conv2DConv2D2gomoku_resnet/expand_17_5x5/Softplus:activations:0;gomoku_resnet/contract_17_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
4gomoku_resnet/contract_17_3x3/BiasAdd/ReadVariableOpReadVariableOp=gomoku_resnet_contract_17_3x3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
%gomoku_resnet/contract_17_3x3/BiasAddBiasAdd-gomoku_resnet/contract_17_3x3/Conv2D:output:0<gomoku_resnet/contract_17_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
&gomoku_resnet/contract_17_3x3/SoftplusSoftplus.gomoku_resnet/contract_17_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:����������
gomoku_resnet/skip_17/addAddV24gomoku_resnet/contract_17_3x3/Softplus:activations:0gomoku_resnet/skip_16/add:z:0*
T0*/
_output_shapes
:���������j
(gomoku_resnet/concatenate_16/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
#gomoku_resnet/concatenate_16/concatConcatV2)gomoku_resnet/heuristic_priority/Tanh:y:0gomoku_resnet/skip_17/add:z:01gomoku_resnet/concatenate_16/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������	�
1gomoku_resnet/expand_18_5x5/Conv2D/ReadVariableOpReadVariableOp:gomoku_resnet_expand_18_5x5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
"gomoku_resnet/expand_18_5x5/Conv2DConv2D,gomoku_resnet/concatenate_16/concat:output:09gomoku_resnet/expand_18_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
2gomoku_resnet/expand_18_5x5/BiasAdd/ReadVariableOpReadVariableOp;gomoku_resnet_expand_18_5x5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
#gomoku_resnet/expand_18_5x5/BiasAddBiasAdd+gomoku_resnet/expand_18_5x5/Conv2D:output:0:gomoku_resnet/expand_18_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
$gomoku_resnet/expand_18_5x5/SoftplusSoftplus,gomoku_resnet/expand_18_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
3gomoku_resnet/contract_18_3x3/Conv2D/ReadVariableOpReadVariableOp<gomoku_resnet_contract_18_3x3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
$gomoku_resnet/contract_18_3x3/Conv2DConv2D2gomoku_resnet/expand_18_5x5/Softplus:activations:0;gomoku_resnet/contract_18_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
4gomoku_resnet/contract_18_3x3/BiasAdd/ReadVariableOpReadVariableOp=gomoku_resnet_contract_18_3x3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
%gomoku_resnet/contract_18_3x3/BiasAddBiasAdd-gomoku_resnet/contract_18_3x3/Conv2D:output:0<gomoku_resnet/contract_18_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
&gomoku_resnet/contract_18_3x3/SoftplusSoftplus.gomoku_resnet/contract_18_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:����������
gomoku_resnet/skip_18/addAddV24gomoku_resnet/contract_18_3x3/Softplus:activations:0gomoku_resnet/skip_17/add:z:0*
T0*/
_output_shapes
:���������j
(gomoku_resnet/concatenate_17/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
#gomoku_resnet/concatenate_17/concatConcatV2)gomoku_resnet/heuristic_priority/Tanh:y:0gomoku_resnet/skip_18/add:z:01gomoku_resnet/concatenate_17/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������	�
1gomoku_resnet/expand_19_5x5/Conv2D/ReadVariableOpReadVariableOp:gomoku_resnet_expand_19_5x5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
"gomoku_resnet/expand_19_5x5/Conv2DConv2D,gomoku_resnet/concatenate_17/concat:output:09gomoku_resnet/expand_19_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
2gomoku_resnet/expand_19_5x5/BiasAdd/ReadVariableOpReadVariableOp;gomoku_resnet_expand_19_5x5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
#gomoku_resnet/expand_19_5x5/BiasAddBiasAdd+gomoku_resnet/expand_19_5x5/Conv2D:output:0:gomoku_resnet/expand_19_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
$gomoku_resnet/expand_19_5x5/SoftplusSoftplus,gomoku_resnet/expand_19_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
3gomoku_resnet/contract_19_3x3/Conv2D/ReadVariableOpReadVariableOp<gomoku_resnet_contract_19_3x3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
$gomoku_resnet/contract_19_3x3/Conv2DConv2D2gomoku_resnet/expand_19_5x5/Softplus:activations:0;gomoku_resnet/contract_19_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
4gomoku_resnet/contract_19_3x3/BiasAdd/ReadVariableOpReadVariableOp=gomoku_resnet_contract_19_3x3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
%gomoku_resnet/contract_19_3x3/BiasAddBiasAdd-gomoku_resnet/contract_19_3x3/Conv2D:output:0<gomoku_resnet/contract_19_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
&gomoku_resnet/contract_19_3x3/SoftplusSoftplus.gomoku_resnet/contract_19_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:����������
gomoku_resnet/skip_19/addAddV24gomoku_resnet/contract_19_3x3/Softplus:activations:0gomoku_resnet/skip_18/add:z:0*
T0*/
_output_shapes
:���������j
(gomoku_resnet/concatenate_18/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
#gomoku_resnet/concatenate_18/concatConcatV2)gomoku_resnet/heuristic_priority/Tanh:y:0gomoku_resnet/skip_19/add:z:01gomoku_resnet/concatenate_18/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������	�
1gomoku_resnet/expand_20_5x5/Conv2D/ReadVariableOpReadVariableOp:gomoku_resnet_expand_20_5x5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
"gomoku_resnet/expand_20_5x5/Conv2DConv2D,gomoku_resnet/concatenate_18/concat:output:09gomoku_resnet/expand_20_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
2gomoku_resnet/expand_20_5x5/BiasAdd/ReadVariableOpReadVariableOp;gomoku_resnet_expand_20_5x5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
#gomoku_resnet/expand_20_5x5/BiasAddBiasAdd+gomoku_resnet/expand_20_5x5/Conv2D:output:0:gomoku_resnet/expand_20_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
$gomoku_resnet/expand_20_5x5/SoftplusSoftplus,gomoku_resnet/expand_20_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
3gomoku_resnet/contract_20_3x3/Conv2D/ReadVariableOpReadVariableOp<gomoku_resnet_contract_20_3x3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
$gomoku_resnet/contract_20_3x3/Conv2DConv2D2gomoku_resnet/expand_20_5x5/Softplus:activations:0;gomoku_resnet/contract_20_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
4gomoku_resnet/contract_20_3x3/BiasAdd/ReadVariableOpReadVariableOp=gomoku_resnet_contract_20_3x3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
%gomoku_resnet/contract_20_3x3/BiasAddBiasAdd-gomoku_resnet/contract_20_3x3/Conv2D:output:0<gomoku_resnet/contract_20_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
&gomoku_resnet/contract_20_3x3/SoftplusSoftplus.gomoku_resnet/contract_20_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:����������
gomoku_resnet/skip_20/addAddV24gomoku_resnet/contract_20_3x3/Softplus:activations:0gomoku_resnet/skip_19/add:z:0*
T0*/
_output_shapes
:���������k
)gomoku_resnet/all_value_input/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
$gomoku_resnet/all_value_input/concatConcatV24gomoku_resnet/contract_20_3x3/Softplus:activations:0)gomoku_resnet/concatenate/concat:output:02gomoku_resnet/all_value_input/concat/axis:output:0*
N*
T0*/
_output_shapes
:����������
5gomoku_resnet/policy_aggregator/Conv2D/ReadVariableOpReadVariableOp>gomoku_resnet_policy_aggregator_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
&gomoku_resnet/policy_aggregator/Conv2DConv2Dgomoku_resnet/skip_20/add:z:0=gomoku_resnet/policy_aggregator/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
6gomoku_resnet/policy_aggregator/BiasAdd/ReadVariableOpReadVariableOp?gomoku_resnet_policy_aggregator_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
'gomoku_resnet/policy_aggregator/BiasAddBiasAdd/gomoku_resnet/policy_aggregator/Conv2D:output:0>gomoku_resnet/policy_aggregator/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
$gomoku_resnet/policy_aggregator/ReluRelu0gomoku_resnet/policy_aggregator/BiasAdd:output:0*
T0*/
_output_shapes
:���������u
$gomoku_resnet/flat_value_input/ConstConst*
_output_shapes
:*
dtype0*
valueB"����e  �
&gomoku_resnet/flat_value_input/ReshapeReshape-gomoku_resnet/all_value_input/concat:output:0-gomoku_resnet/flat_value_input/Const:output:0*
T0*(
_output_shapes
:����������,�
.gomoku_resnet/border_off/Conv2D/ReadVariableOpReadVariableOp7gomoku_resnet_border_off_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
gomoku_resnet/border_off/Conv2DConv2D2gomoku_resnet/policy_aggregator/Relu:activations:06gomoku_resnet/border_off/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
/gomoku_resnet/border_off/BiasAdd/ReadVariableOpReadVariableOp8gomoku_resnet_border_off_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
 gomoku_resnet/border_off/BiasAddBiasAdd(gomoku_resnet/border_off/Conv2D:output:07gomoku_resnet/border_off/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������l
'gomoku_resnet/tf.math.truediv/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �B�
%gomoku_resnet/tf.math.truediv/truedivRealDiv/gomoku_resnet/flat_value_input/Reshape:output:00gomoku_resnet/tf.math.truediv/truediv/y:output:0*
T0*(
_output_shapes
:����������,p
gomoku_resnet/flat_logits/ConstConst*
_output_shapes
:*
dtype0*
valueB"����i  �
!gomoku_resnet/flat_logits/ReshapeReshape)gomoku_resnet/border_off/BiasAdd:output:0(gomoku_resnet/flat_logits/Const:output:0*
T0*(
_output_shapes
:�����������
.gomoku_resnet/value_head/MatMul/ReadVariableOpReadVariableOp7gomoku_resnet_value_head_matmul_readvariableop_resource*
_output_shapes
:	�,*
dtype0�
gomoku_resnet/value_head/MatMulMatMul)gomoku_resnet/tf.math.truediv/truediv:z:06gomoku_resnet/value_head/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
/gomoku_resnet/value_head/BiasAdd/ReadVariableOpReadVariableOp8gomoku_resnet_value_head_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
 gomoku_resnet/value_head/BiasAddBiasAdd)gomoku_resnet/value_head/MatMul:product:07gomoku_resnet/value_head/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
gomoku_resnet/value_head/TanhTanh)gomoku_resnet/value_head/BiasAdd:output:0*
T0*'
_output_shapes
:����������
!gomoku_resnet/policy_head/SoftmaxSoftmax*gomoku_resnet/flat_logits/Reshape:output:0*
T0*(
_output_shapes
:����������{
IdentityIdentity+gomoku_resnet/policy_head/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:����������r

Identity_1Identity!gomoku_resnet/value_head/Tanh:y:0^NoOp*
T0*'
_output_shapes
:����������%
NoOpNoOp0^gomoku_resnet/border_off/BiasAdd/ReadVariableOp/^gomoku_resnet/border_off/Conv2D/ReadVariableOp5^gomoku_resnet/contract_10_3x3/BiasAdd/ReadVariableOp4^gomoku_resnet/contract_10_3x3/Conv2D/ReadVariableOp5^gomoku_resnet/contract_11_3x3/BiasAdd/ReadVariableOp4^gomoku_resnet/contract_11_3x3/Conv2D/ReadVariableOp5^gomoku_resnet/contract_12_3x3/BiasAdd/ReadVariableOp4^gomoku_resnet/contract_12_3x3/Conv2D/ReadVariableOp5^gomoku_resnet/contract_13_3x3/BiasAdd/ReadVariableOp4^gomoku_resnet/contract_13_3x3/Conv2D/ReadVariableOp5^gomoku_resnet/contract_14_3x3/BiasAdd/ReadVariableOp4^gomoku_resnet/contract_14_3x3/Conv2D/ReadVariableOp5^gomoku_resnet/contract_15_3x3/BiasAdd/ReadVariableOp4^gomoku_resnet/contract_15_3x3/Conv2D/ReadVariableOp5^gomoku_resnet/contract_16_3x3/BiasAdd/ReadVariableOp4^gomoku_resnet/contract_16_3x3/Conv2D/ReadVariableOp5^gomoku_resnet/contract_17_3x3/BiasAdd/ReadVariableOp4^gomoku_resnet/contract_17_3x3/Conv2D/ReadVariableOp5^gomoku_resnet/contract_18_3x3/BiasAdd/ReadVariableOp4^gomoku_resnet/contract_18_3x3/Conv2D/ReadVariableOp5^gomoku_resnet/contract_19_3x3/BiasAdd/ReadVariableOp4^gomoku_resnet/contract_19_3x3/Conv2D/ReadVariableOp4^gomoku_resnet/contract_1_5x5/BiasAdd/ReadVariableOp3^gomoku_resnet/contract_1_5x5/Conv2D/ReadVariableOp5^gomoku_resnet/contract_20_3x3/BiasAdd/ReadVariableOp4^gomoku_resnet/contract_20_3x3/Conv2D/ReadVariableOp4^gomoku_resnet/contract_2_3x3/BiasAdd/ReadVariableOp3^gomoku_resnet/contract_2_3x3/Conv2D/ReadVariableOp4^gomoku_resnet/contract_3_3x3/BiasAdd/ReadVariableOp3^gomoku_resnet/contract_3_3x3/Conv2D/ReadVariableOp4^gomoku_resnet/contract_4_3x3/BiasAdd/ReadVariableOp3^gomoku_resnet/contract_4_3x3/Conv2D/ReadVariableOp4^gomoku_resnet/contract_5_3x3/BiasAdd/ReadVariableOp3^gomoku_resnet/contract_5_3x3/Conv2D/ReadVariableOp4^gomoku_resnet/contract_6_3x3/BiasAdd/ReadVariableOp3^gomoku_resnet/contract_6_3x3/Conv2D/ReadVariableOp4^gomoku_resnet/contract_7_3x3/BiasAdd/ReadVariableOp3^gomoku_resnet/contract_7_3x3/Conv2D/ReadVariableOp4^gomoku_resnet/contract_8_3x3/BiasAdd/ReadVariableOp3^gomoku_resnet/contract_8_3x3/Conv2D/ReadVariableOp4^gomoku_resnet/contract_9_3x3/BiasAdd/ReadVariableOp3^gomoku_resnet/contract_9_3x3/Conv2D/ReadVariableOp3^gomoku_resnet/expand_10_5x5/BiasAdd/ReadVariableOp2^gomoku_resnet/expand_10_5x5/Conv2D/ReadVariableOp3^gomoku_resnet/expand_11_5x5/BiasAdd/ReadVariableOp2^gomoku_resnet/expand_11_5x5/Conv2D/ReadVariableOp3^gomoku_resnet/expand_12_5x5/BiasAdd/ReadVariableOp2^gomoku_resnet/expand_12_5x5/Conv2D/ReadVariableOp3^gomoku_resnet/expand_13_5x5/BiasAdd/ReadVariableOp2^gomoku_resnet/expand_13_5x5/Conv2D/ReadVariableOp3^gomoku_resnet/expand_14_5x5/BiasAdd/ReadVariableOp2^gomoku_resnet/expand_14_5x5/Conv2D/ReadVariableOp3^gomoku_resnet/expand_15_5x5/BiasAdd/ReadVariableOp2^gomoku_resnet/expand_15_5x5/Conv2D/ReadVariableOp3^gomoku_resnet/expand_16_5x5/BiasAdd/ReadVariableOp2^gomoku_resnet/expand_16_5x5/Conv2D/ReadVariableOp3^gomoku_resnet/expand_17_5x5/BiasAdd/ReadVariableOp2^gomoku_resnet/expand_17_5x5/Conv2D/ReadVariableOp3^gomoku_resnet/expand_18_5x5/BiasAdd/ReadVariableOp2^gomoku_resnet/expand_18_5x5/Conv2D/ReadVariableOp3^gomoku_resnet/expand_19_5x5/BiasAdd/ReadVariableOp2^gomoku_resnet/expand_19_5x5/Conv2D/ReadVariableOp4^gomoku_resnet/expand_1_11x11/BiasAdd/ReadVariableOp3^gomoku_resnet/expand_1_11x11/Conv2D/ReadVariableOp3^gomoku_resnet/expand_20_5x5/BiasAdd/ReadVariableOp2^gomoku_resnet/expand_20_5x5/Conv2D/ReadVariableOp2^gomoku_resnet/expand_2_5x5/BiasAdd/ReadVariableOp1^gomoku_resnet/expand_2_5x5/Conv2D/ReadVariableOp2^gomoku_resnet/expand_3_5x5/BiasAdd/ReadVariableOp1^gomoku_resnet/expand_3_5x5/Conv2D/ReadVariableOp2^gomoku_resnet/expand_4_5x5/BiasAdd/ReadVariableOp1^gomoku_resnet/expand_4_5x5/Conv2D/ReadVariableOp2^gomoku_resnet/expand_5_5x5/BiasAdd/ReadVariableOp1^gomoku_resnet/expand_5_5x5/Conv2D/ReadVariableOp2^gomoku_resnet/expand_6_5x5/BiasAdd/ReadVariableOp1^gomoku_resnet/expand_6_5x5/Conv2D/ReadVariableOp2^gomoku_resnet/expand_7_5x5/BiasAdd/ReadVariableOp1^gomoku_resnet/expand_7_5x5/Conv2D/ReadVariableOp2^gomoku_resnet/expand_8_5x5/BiasAdd/ReadVariableOp1^gomoku_resnet/expand_8_5x5/Conv2D/ReadVariableOp2^gomoku_resnet/expand_9_5x5/BiasAdd/ReadVariableOp1^gomoku_resnet/expand_9_5x5/Conv2D/ReadVariableOp8^gomoku_resnet/heuristic_detector/BiasAdd/ReadVariableOp7^gomoku_resnet/heuristic_detector/Conv2D/ReadVariableOp8^gomoku_resnet/heuristic_priority/BiasAdd/ReadVariableOp7^gomoku_resnet/heuristic_priority/Conv2D/ReadVariableOp7^gomoku_resnet/policy_aggregator/BiasAdd/ReadVariableOp6^gomoku_resnet/policy_aggregator/Conv2D/ReadVariableOp0^gomoku_resnet/value_head/BiasAdd/ReadVariableOp/^gomoku_resnet/value_head/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/gomoku_resnet/border_off/BiasAdd/ReadVariableOp/gomoku_resnet/border_off/BiasAdd/ReadVariableOp2`
.gomoku_resnet/border_off/Conv2D/ReadVariableOp.gomoku_resnet/border_off/Conv2D/ReadVariableOp2l
4gomoku_resnet/contract_10_3x3/BiasAdd/ReadVariableOp4gomoku_resnet/contract_10_3x3/BiasAdd/ReadVariableOp2j
3gomoku_resnet/contract_10_3x3/Conv2D/ReadVariableOp3gomoku_resnet/contract_10_3x3/Conv2D/ReadVariableOp2l
4gomoku_resnet/contract_11_3x3/BiasAdd/ReadVariableOp4gomoku_resnet/contract_11_3x3/BiasAdd/ReadVariableOp2j
3gomoku_resnet/contract_11_3x3/Conv2D/ReadVariableOp3gomoku_resnet/contract_11_3x3/Conv2D/ReadVariableOp2l
4gomoku_resnet/contract_12_3x3/BiasAdd/ReadVariableOp4gomoku_resnet/contract_12_3x3/BiasAdd/ReadVariableOp2j
3gomoku_resnet/contract_12_3x3/Conv2D/ReadVariableOp3gomoku_resnet/contract_12_3x3/Conv2D/ReadVariableOp2l
4gomoku_resnet/contract_13_3x3/BiasAdd/ReadVariableOp4gomoku_resnet/contract_13_3x3/BiasAdd/ReadVariableOp2j
3gomoku_resnet/contract_13_3x3/Conv2D/ReadVariableOp3gomoku_resnet/contract_13_3x3/Conv2D/ReadVariableOp2l
4gomoku_resnet/contract_14_3x3/BiasAdd/ReadVariableOp4gomoku_resnet/contract_14_3x3/BiasAdd/ReadVariableOp2j
3gomoku_resnet/contract_14_3x3/Conv2D/ReadVariableOp3gomoku_resnet/contract_14_3x3/Conv2D/ReadVariableOp2l
4gomoku_resnet/contract_15_3x3/BiasAdd/ReadVariableOp4gomoku_resnet/contract_15_3x3/BiasAdd/ReadVariableOp2j
3gomoku_resnet/contract_15_3x3/Conv2D/ReadVariableOp3gomoku_resnet/contract_15_3x3/Conv2D/ReadVariableOp2l
4gomoku_resnet/contract_16_3x3/BiasAdd/ReadVariableOp4gomoku_resnet/contract_16_3x3/BiasAdd/ReadVariableOp2j
3gomoku_resnet/contract_16_3x3/Conv2D/ReadVariableOp3gomoku_resnet/contract_16_3x3/Conv2D/ReadVariableOp2l
4gomoku_resnet/contract_17_3x3/BiasAdd/ReadVariableOp4gomoku_resnet/contract_17_3x3/BiasAdd/ReadVariableOp2j
3gomoku_resnet/contract_17_3x3/Conv2D/ReadVariableOp3gomoku_resnet/contract_17_3x3/Conv2D/ReadVariableOp2l
4gomoku_resnet/contract_18_3x3/BiasAdd/ReadVariableOp4gomoku_resnet/contract_18_3x3/BiasAdd/ReadVariableOp2j
3gomoku_resnet/contract_18_3x3/Conv2D/ReadVariableOp3gomoku_resnet/contract_18_3x3/Conv2D/ReadVariableOp2l
4gomoku_resnet/contract_19_3x3/BiasAdd/ReadVariableOp4gomoku_resnet/contract_19_3x3/BiasAdd/ReadVariableOp2j
3gomoku_resnet/contract_19_3x3/Conv2D/ReadVariableOp3gomoku_resnet/contract_19_3x3/Conv2D/ReadVariableOp2j
3gomoku_resnet/contract_1_5x5/BiasAdd/ReadVariableOp3gomoku_resnet/contract_1_5x5/BiasAdd/ReadVariableOp2h
2gomoku_resnet/contract_1_5x5/Conv2D/ReadVariableOp2gomoku_resnet/contract_1_5x5/Conv2D/ReadVariableOp2l
4gomoku_resnet/contract_20_3x3/BiasAdd/ReadVariableOp4gomoku_resnet/contract_20_3x3/BiasAdd/ReadVariableOp2j
3gomoku_resnet/contract_20_3x3/Conv2D/ReadVariableOp3gomoku_resnet/contract_20_3x3/Conv2D/ReadVariableOp2j
3gomoku_resnet/contract_2_3x3/BiasAdd/ReadVariableOp3gomoku_resnet/contract_2_3x3/BiasAdd/ReadVariableOp2h
2gomoku_resnet/contract_2_3x3/Conv2D/ReadVariableOp2gomoku_resnet/contract_2_3x3/Conv2D/ReadVariableOp2j
3gomoku_resnet/contract_3_3x3/BiasAdd/ReadVariableOp3gomoku_resnet/contract_3_3x3/BiasAdd/ReadVariableOp2h
2gomoku_resnet/contract_3_3x3/Conv2D/ReadVariableOp2gomoku_resnet/contract_3_3x3/Conv2D/ReadVariableOp2j
3gomoku_resnet/contract_4_3x3/BiasAdd/ReadVariableOp3gomoku_resnet/contract_4_3x3/BiasAdd/ReadVariableOp2h
2gomoku_resnet/contract_4_3x3/Conv2D/ReadVariableOp2gomoku_resnet/contract_4_3x3/Conv2D/ReadVariableOp2j
3gomoku_resnet/contract_5_3x3/BiasAdd/ReadVariableOp3gomoku_resnet/contract_5_3x3/BiasAdd/ReadVariableOp2h
2gomoku_resnet/contract_5_3x3/Conv2D/ReadVariableOp2gomoku_resnet/contract_5_3x3/Conv2D/ReadVariableOp2j
3gomoku_resnet/contract_6_3x3/BiasAdd/ReadVariableOp3gomoku_resnet/contract_6_3x3/BiasAdd/ReadVariableOp2h
2gomoku_resnet/contract_6_3x3/Conv2D/ReadVariableOp2gomoku_resnet/contract_6_3x3/Conv2D/ReadVariableOp2j
3gomoku_resnet/contract_7_3x3/BiasAdd/ReadVariableOp3gomoku_resnet/contract_7_3x3/BiasAdd/ReadVariableOp2h
2gomoku_resnet/contract_7_3x3/Conv2D/ReadVariableOp2gomoku_resnet/contract_7_3x3/Conv2D/ReadVariableOp2j
3gomoku_resnet/contract_8_3x3/BiasAdd/ReadVariableOp3gomoku_resnet/contract_8_3x3/BiasAdd/ReadVariableOp2h
2gomoku_resnet/contract_8_3x3/Conv2D/ReadVariableOp2gomoku_resnet/contract_8_3x3/Conv2D/ReadVariableOp2j
3gomoku_resnet/contract_9_3x3/BiasAdd/ReadVariableOp3gomoku_resnet/contract_9_3x3/BiasAdd/ReadVariableOp2h
2gomoku_resnet/contract_9_3x3/Conv2D/ReadVariableOp2gomoku_resnet/contract_9_3x3/Conv2D/ReadVariableOp2h
2gomoku_resnet/expand_10_5x5/BiasAdd/ReadVariableOp2gomoku_resnet/expand_10_5x5/BiasAdd/ReadVariableOp2f
1gomoku_resnet/expand_10_5x5/Conv2D/ReadVariableOp1gomoku_resnet/expand_10_5x5/Conv2D/ReadVariableOp2h
2gomoku_resnet/expand_11_5x5/BiasAdd/ReadVariableOp2gomoku_resnet/expand_11_5x5/BiasAdd/ReadVariableOp2f
1gomoku_resnet/expand_11_5x5/Conv2D/ReadVariableOp1gomoku_resnet/expand_11_5x5/Conv2D/ReadVariableOp2h
2gomoku_resnet/expand_12_5x5/BiasAdd/ReadVariableOp2gomoku_resnet/expand_12_5x5/BiasAdd/ReadVariableOp2f
1gomoku_resnet/expand_12_5x5/Conv2D/ReadVariableOp1gomoku_resnet/expand_12_5x5/Conv2D/ReadVariableOp2h
2gomoku_resnet/expand_13_5x5/BiasAdd/ReadVariableOp2gomoku_resnet/expand_13_5x5/BiasAdd/ReadVariableOp2f
1gomoku_resnet/expand_13_5x5/Conv2D/ReadVariableOp1gomoku_resnet/expand_13_5x5/Conv2D/ReadVariableOp2h
2gomoku_resnet/expand_14_5x5/BiasAdd/ReadVariableOp2gomoku_resnet/expand_14_5x5/BiasAdd/ReadVariableOp2f
1gomoku_resnet/expand_14_5x5/Conv2D/ReadVariableOp1gomoku_resnet/expand_14_5x5/Conv2D/ReadVariableOp2h
2gomoku_resnet/expand_15_5x5/BiasAdd/ReadVariableOp2gomoku_resnet/expand_15_5x5/BiasAdd/ReadVariableOp2f
1gomoku_resnet/expand_15_5x5/Conv2D/ReadVariableOp1gomoku_resnet/expand_15_5x5/Conv2D/ReadVariableOp2h
2gomoku_resnet/expand_16_5x5/BiasAdd/ReadVariableOp2gomoku_resnet/expand_16_5x5/BiasAdd/ReadVariableOp2f
1gomoku_resnet/expand_16_5x5/Conv2D/ReadVariableOp1gomoku_resnet/expand_16_5x5/Conv2D/ReadVariableOp2h
2gomoku_resnet/expand_17_5x5/BiasAdd/ReadVariableOp2gomoku_resnet/expand_17_5x5/BiasAdd/ReadVariableOp2f
1gomoku_resnet/expand_17_5x5/Conv2D/ReadVariableOp1gomoku_resnet/expand_17_5x5/Conv2D/ReadVariableOp2h
2gomoku_resnet/expand_18_5x5/BiasAdd/ReadVariableOp2gomoku_resnet/expand_18_5x5/BiasAdd/ReadVariableOp2f
1gomoku_resnet/expand_18_5x5/Conv2D/ReadVariableOp1gomoku_resnet/expand_18_5x5/Conv2D/ReadVariableOp2h
2gomoku_resnet/expand_19_5x5/BiasAdd/ReadVariableOp2gomoku_resnet/expand_19_5x5/BiasAdd/ReadVariableOp2f
1gomoku_resnet/expand_19_5x5/Conv2D/ReadVariableOp1gomoku_resnet/expand_19_5x5/Conv2D/ReadVariableOp2j
3gomoku_resnet/expand_1_11x11/BiasAdd/ReadVariableOp3gomoku_resnet/expand_1_11x11/BiasAdd/ReadVariableOp2h
2gomoku_resnet/expand_1_11x11/Conv2D/ReadVariableOp2gomoku_resnet/expand_1_11x11/Conv2D/ReadVariableOp2h
2gomoku_resnet/expand_20_5x5/BiasAdd/ReadVariableOp2gomoku_resnet/expand_20_5x5/BiasAdd/ReadVariableOp2f
1gomoku_resnet/expand_20_5x5/Conv2D/ReadVariableOp1gomoku_resnet/expand_20_5x5/Conv2D/ReadVariableOp2f
1gomoku_resnet/expand_2_5x5/BiasAdd/ReadVariableOp1gomoku_resnet/expand_2_5x5/BiasAdd/ReadVariableOp2d
0gomoku_resnet/expand_2_5x5/Conv2D/ReadVariableOp0gomoku_resnet/expand_2_5x5/Conv2D/ReadVariableOp2f
1gomoku_resnet/expand_3_5x5/BiasAdd/ReadVariableOp1gomoku_resnet/expand_3_5x5/BiasAdd/ReadVariableOp2d
0gomoku_resnet/expand_3_5x5/Conv2D/ReadVariableOp0gomoku_resnet/expand_3_5x5/Conv2D/ReadVariableOp2f
1gomoku_resnet/expand_4_5x5/BiasAdd/ReadVariableOp1gomoku_resnet/expand_4_5x5/BiasAdd/ReadVariableOp2d
0gomoku_resnet/expand_4_5x5/Conv2D/ReadVariableOp0gomoku_resnet/expand_4_5x5/Conv2D/ReadVariableOp2f
1gomoku_resnet/expand_5_5x5/BiasAdd/ReadVariableOp1gomoku_resnet/expand_5_5x5/BiasAdd/ReadVariableOp2d
0gomoku_resnet/expand_5_5x5/Conv2D/ReadVariableOp0gomoku_resnet/expand_5_5x5/Conv2D/ReadVariableOp2f
1gomoku_resnet/expand_6_5x5/BiasAdd/ReadVariableOp1gomoku_resnet/expand_6_5x5/BiasAdd/ReadVariableOp2d
0gomoku_resnet/expand_6_5x5/Conv2D/ReadVariableOp0gomoku_resnet/expand_6_5x5/Conv2D/ReadVariableOp2f
1gomoku_resnet/expand_7_5x5/BiasAdd/ReadVariableOp1gomoku_resnet/expand_7_5x5/BiasAdd/ReadVariableOp2d
0gomoku_resnet/expand_7_5x5/Conv2D/ReadVariableOp0gomoku_resnet/expand_7_5x5/Conv2D/ReadVariableOp2f
1gomoku_resnet/expand_8_5x5/BiasAdd/ReadVariableOp1gomoku_resnet/expand_8_5x5/BiasAdd/ReadVariableOp2d
0gomoku_resnet/expand_8_5x5/Conv2D/ReadVariableOp0gomoku_resnet/expand_8_5x5/Conv2D/ReadVariableOp2f
1gomoku_resnet/expand_9_5x5/BiasAdd/ReadVariableOp1gomoku_resnet/expand_9_5x5/BiasAdd/ReadVariableOp2d
0gomoku_resnet/expand_9_5x5/Conv2D/ReadVariableOp0gomoku_resnet/expand_9_5x5/Conv2D/ReadVariableOp2r
7gomoku_resnet/heuristic_detector/BiasAdd/ReadVariableOp7gomoku_resnet/heuristic_detector/BiasAdd/ReadVariableOp2p
6gomoku_resnet/heuristic_detector/Conv2D/ReadVariableOp6gomoku_resnet/heuristic_detector/Conv2D/ReadVariableOp2r
7gomoku_resnet/heuristic_priority/BiasAdd/ReadVariableOp7gomoku_resnet/heuristic_priority/BiasAdd/ReadVariableOp2p
6gomoku_resnet/heuristic_priority/Conv2D/ReadVariableOp6gomoku_resnet/heuristic_priority/Conv2D/ReadVariableOp2p
6gomoku_resnet/policy_aggregator/BiasAdd/ReadVariableOp6gomoku_resnet/policy_aggregator/BiasAdd/ReadVariableOp2n
5gomoku_resnet/policy_aggregator/Conv2D/ReadVariableOp5gomoku_resnet/policy_aggregator/Conv2D/ReadVariableOp2b
/gomoku_resnet/value_head/BiasAdd/ReadVariableOp/gomoku_resnet/value_head/BiasAdd/ReadVariableOp2`
.gomoku_resnet/value_head/MatMul/ReadVariableOp.gomoku_resnet/value_head/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
u
K__inference_concatenate_10_layer_call_and_return_conditional_losses_3614821

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
:���������	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
v
J__inference_concatenate_1_layer_call_and_return_conditional_losses_3618015
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
:���������	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
Ё
�-
J__inference_gomoku_resnet_layer_call_and_return_conditional_losses_3617668

inputs1
expand_1_11x11_3617397:�%
expand_1_11x11_3617399:	�5
heuristic_detector_3617402:�)
heuristic_detector_3617404:	�5
heuristic_priority_3617407:�(
heuristic_priority_3617409:1
contract_1_5x5_3617412:�$
contract_1_5x5_3617414:.
expand_2_5x5_3617418:	 "
expand_2_5x5_3617420: 0
contract_2_3x3_3617423: $
contract_2_3x3_3617425:.
expand_3_5x5_3617430:	 "
expand_3_5x5_3617432: 0
contract_3_3x3_3617435: $
contract_3_3x3_3617437:.
expand_4_5x5_3617442:	 "
expand_4_5x5_3617444: 0
contract_4_3x3_3617447: $
contract_4_3x3_3617449:.
expand_5_5x5_3617454:	 "
expand_5_5x5_3617456: 0
contract_5_3x3_3617459: $
contract_5_3x3_3617461:.
expand_6_5x5_3617466:	 "
expand_6_5x5_3617468: 0
contract_6_3x3_3617471: $
contract_6_3x3_3617473:.
expand_7_5x5_3617478:	 "
expand_7_5x5_3617480: 0
contract_7_3x3_3617483: $
contract_7_3x3_3617485:.
expand_8_5x5_3617490:	 "
expand_8_5x5_3617492: 0
contract_8_3x3_3617495: $
contract_8_3x3_3617497:.
expand_9_5x5_3617502:	 "
expand_9_5x5_3617504: 0
contract_9_3x3_3617507: $
contract_9_3x3_3617509:/
expand_10_5x5_3617514:	 #
expand_10_5x5_3617516: 1
contract_10_3x3_3617519: %
contract_10_3x3_3617521:/
expand_11_5x5_3617526:	 #
expand_11_5x5_3617528: 1
contract_11_3x3_3617531: %
contract_11_3x3_3617533:/
expand_12_5x5_3617538:	 #
expand_12_5x5_3617540: 1
contract_12_3x3_3617543: %
contract_12_3x3_3617545:/
expand_13_5x5_3617550:	 #
expand_13_5x5_3617552: 1
contract_13_3x3_3617555: %
contract_13_3x3_3617557:/
expand_14_5x5_3617562:	 #
expand_14_5x5_3617564: 1
contract_14_3x3_3617567: %
contract_14_3x3_3617569:/
expand_15_5x5_3617574:	 #
expand_15_5x5_3617576: 1
contract_15_3x3_3617579: %
contract_15_3x3_3617581:/
expand_16_5x5_3617586:	 #
expand_16_5x5_3617588: 1
contract_16_3x3_3617591: %
contract_16_3x3_3617593:/
expand_17_5x5_3617598:	 #
expand_17_5x5_3617600: 1
contract_17_3x3_3617603: %
contract_17_3x3_3617605:/
expand_18_5x5_3617610:	 #
expand_18_5x5_3617612: 1
contract_18_3x3_3617615: %
contract_18_3x3_3617617:/
expand_19_5x5_3617622:	 #
expand_19_5x5_3617624: 1
contract_19_3x3_3617627: %
contract_19_3x3_3617629:/
expand_20_5x5_3617634:	 #
expand_20_5x5_3617636: 1
contract_20_3x3_3617639: %
contract_20_3x3_3617641:3
policy_aggregator_3617646:'
policy_aggregator_3617648:,
border_off_3617652: 
border_off_3617654:%
value_head_3617660:	�, 
value_head_3617662:
identity

identity_1��"border_off/StatefulPartitionedCall�'contract_10_3x3/StatefulPartitionedCall�'contract_11_3x3/StatefulPartitionedCall�'contract_12_3x3/StatefulPartitionedCall�'contract_13_3x3/StatefulPartitionedCall�'contract_14_3x3/StatefulPartitionedCall�'contract_15_3x3/StatefulPartitionedCall�'contract_16_3x3/StatefulPartitionedCall�'contract_17_3x3/StatefulPartitionedCall�'contract_18_3x3/StatefulPartitionedCall�'contract_19_3x3/StatefulPartitionedCall�&contract_1_5x5/StatefulPartitionedCall�'contract_20_3x3/StatefulPartitionedCall�&contract_2_3x3/StatefulPartitionedCall�&contract_3_3x3/StatefulPartitionedCall�&contract_4_3x3/StatefulPartitionedCall�&contract_5_3x3/StatefulPartitionedCall�&contract_6_3x3/StatefulPartitionedCall�&contract_7_3x3/StatefulPartitionedCall�&contract_8_3x3/StatefulPartitionedCall�&contract_9_3x3/StatefulPartitionedCall�%expand_10_5x5/StatefulPartitionedCall�%expand_11_5x5/StatefulPartitionedCall�%expand_12_5x5/StatefulPartitionedCall�%expand_13_5x5/StatefulPartitionedCall�%expand_14_5x5/StatefulPartitionedCall�%expand_15_5x5/StatefulPartitionedCall�%expand_16_5x5/StatefulPartitionedCall�%expand_17_5x5/StatefulPartitionedCall�%expand_18_5x5/StatefulPartitionedCall�%expand_19_5x5/StatefulPartitionedCall�&expand_1_11x11/StatefulPartitionedCall�%expand_20_5x5/StatefulPartitionedCall�$expand_2_5x5/StatefulPartitionedCall�$expand_3_5x5/StatefulPartitionedCall�$expand_4_5x5/StatefulPartitionedCall�$expand_5_5x5/StatefulPartitionedCall�$expand_6_5x5/StatefulPartitionedCall�$expand_7_5x5/StatefulPartitionedCall�$expand_8_5x5/StatefulPartitionedCall�$expand_9_5x5/StatefulPartitionedCall�*heuristic_detector/StatefulPartitionedCall�*heuristic_priority/StatefulPartitionedCall�)policy_aggregator/StatefulPartitionedCall�"value_head/StatefulPartitionedCall�
&expand_1_11x11/StatefulPartitionedCallStatefulPartitionedCallinputsexpand_1_11x11_3617397expand_1_11x11_3617399*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_expand_1_11x11_layer_call_and_return_conditional_losses_3614247�
*heuristic_detector/StatefulPartitionedCallStatefulPartitionedCallinputsheuristic_detector_3617402heuristic_detector_3617404*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_heuristic_detector_layer_call_and_return_conditional_losses_3614264�
*heuristic_priority/StatefulPartitionedCallStatefulPartitionedCall3heuristic_detector/StatefulPartitionedCall:output:0heuristic_priority_3617407heuristic_priority_3617409*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_heuristic_priority_layer_call_and_return_conditional_losses_3614281�
&contract_1_5x5/StatefulPartitionedCallStatefulPartitionedCall/expand_1_11x11/StatefulPartitionedCall:output:0contract_1_5x5_3617412contract_1_5x5_3617414*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_1_5x5_layer_call_and_return_conditional_losses_3614298�
concatenate/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0/contract_1_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_3614311�
$expand_2_5x5/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0expand_2_5x5_3617418expand_2_5x5_3617420*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_2_5x5_layer_call_and_return_conditional_losses_3614324�
&contract_2_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_2_5x5/StatefulPartitionedCall:output:0contract_2_3x3_3617423contract_2_3x3_3617425*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_2_3x3_layer_call_and_return_conditional_losses_3614341�
skip_2/PartitionedCallPartitionedCall/contract_2_3x3/StatefulPartitionedCall:output:0/contract_1_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_2_layer_call_and_return_conditional_losses_3614353�
concatenate_1/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_3614362�
$expand_3_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0expand_3_5x5_3617430expand_3_5x5_3617432*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_3_5x5_layer_call_and_return_conditional_losses_3614375�
&contract_3_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_3_5x5/StatefulPartitionedCall:output:0contract_3_3x3_3617435contract_3_3x3_3617437*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_3_3x3_layer_call_and_return_conditional_losses_3614392�
skip_3/PartitionedCallPartitionedCall/contract_3_3x3/StatefulPartitionedCall:output:0skip_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_3_layer_call_and_return_conditional_losses_3614404�
concatenate_2/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_3614413�
$expand_4_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0expand_4_5x5_3617442expand_4_5x5_3617444*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_4_5x5_layer_call_and_return_conditional_losses_3614426�
&contract_4_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_4_5x5/StatefulPartitionedCall:output:0contract_4_3x3_3617447contract_4_3x3_3617449*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_4_3x3_layer_call_and_return_conditional_losses_3614443�
skip_4/PartitionedCallPartitionedCall/contract_4_3x3/StatefulPartitionedCall:output:0skip_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_4_layer_call_and_return_conditional_losses_3614455�
concatenate_3/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_3_layer_call_and_return_conditional_losses_3614464�
$expand_5_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0expand_5_5x5_3617454expand_5_5x5_3617456*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_5_5x5_layer_call_and_return_conditional_losses_3614477�
&contract_5_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_5_5x5/StatefulPartitionedCall:output:0contract_5_3x3_3617459contract_5_3x3_3617461*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_5_3x3_layer_call_and_return_conditional_losses_3614494�
skip_5/PartitionedCallPartitionedCall/contract_5_3x3/StatefulPartitionedCall:output:0skip_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_5_layer_call_and_return_conditional_losses_3614506�
concatenate_4/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_4_layer_call_and_return_conditional_losses_3614515�
$expand_6_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0expand_6_5x5_3617466expand_6_5x5_3617468*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_6_5x5_layer_call_and_return_conditional_losses_3614528�
&contract_6_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_6_5x5/StatefulPartitionedCall:output:0contract_6_3x3_3617471contract_6_3x3_3617473*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_6_3x3_layer_call_and_return_conditional_losses_3614545�
skip_6/PartitionedCallPartitionedCall/contract_6_3x3/StatefulPartitionedCall:output:0skip_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_6_layer_call_and_return_conditional_losses_3614557�
concatenate_5/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_5_layer_call_and_return_conditional_losses_3614566�
$expand_7_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0expand_7_5x5_3617478expand_7_5x5_3617480*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_7_5x5_layer_call_and_return_conditional_losses_3614579�
&contract_7_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_7_5x5/StatefulPartitionedCall:output:0contract_7_3x3_3617483contract_7_3x3_3617485*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_7_3x3_layer_call_and_return_conditional_losses_3614596�
skip_7/PartitionedCallPartitionedCall/contract_7_3x3/StatefulPartitionedCall:output:0skip_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_7_layer_call_and_return_conditional_losses_3614608�
concatenate_6/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_6_layer_call_and_return_conditional_losses_3614617�
$expand_8_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_6/PartitionedCall:output:0expand_8_5x5_3617490expand_8_5x5_3617492*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_8_5x5_layer_call_and_return_conditional_losses_3614630�
&contract_8_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_8_5x5/StatefulPartitionedCall:output:0contract_8_3x3_3617495contract_8_3x3_3617497*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_8_3x3_layer_call_and_return_conditional_losses_3614647�
skip_8/PartitionedCallPartitionedCall/contract_8_3x3/StatefulPartitionedCall:output:0skip_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_8_layer_call_and_return_conditional_losses_3614659�
concatenate_7/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_7_layer_call_and_return_conditional_losses_3614668�
$expand_9_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_7/PartitionedCall:output:0expand_9_5x5_3617502expand_9_5x5_3617504*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_9_5x5_layer_call_and_return_conditional_losses_3614681�
&contract_9_3x3/StatefulPartitionedCallStatefulPartitionedCall-expand_9_5x5/StatefulPartitionedCall:output:0contract_9_3x3_3617507contract_9_3x3_3617509*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_9_3x3_layer_call_and_return_conditional_losses_3614698�
skip_9/PartitionedCallPartitionedCall/contract_9_3x3/StatefulPartitionedCall:output:0skip_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_skip_9_layer_call_and_return_conditional_losses_3614710�
concatenate_8/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0skip_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_8_layer_call_and_return_conditional_losses_3614719�
%expand_10_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_8/PartitionedCall:output:0expand_10_5x5_3617514expand_10_5x5_3617516*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_10_5x5_layer_call_and_return_conditional_losses_3614732�
'contract_10_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_10_5x5/StatefulPartitionedCall:output:0contract_10_3x3_3617519contract_10_3x3_3617521*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_10_3x3_layer_call_and_return_conditional_losses_3614749�
skip_10/PartitionedCallPartitionedCall0contract_10_3x3/StatefulPartitionedCall:output:0skip_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_10_layer_call_and_return_conditional_losses_3614761�
concatenate_9/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_9_layer_call_and_return_conditional_losses_3614770�
%expand_11_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_9/PartitionedCall:output:0expand_11_5x5_3617526expand_11_5x5_3617528*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_11_5x5_layer_call_and_return_conditional_losses_3614783�
'contract_11_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_11_5x5/StatefulPartitionedCall:output:0contract_11_3x3_3617531contract_11_3x3_3617533*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_11_3x3_layer_call_and_return_conditional_losses_3614800�
skip_11/PartitionedCallPartitionedCall0contract_11_3x3/StatefulPartitionedCall:output:0 skip_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_11_layer_call_and_return_conditional_losses_3614812�
concatenate_10/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_10_layer_call_and_return_conditional_losses_3614821�
%expand_12_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_10/PartitionedCall:output:0expand_12_5x5_3617538expand_12_5x5_3617540*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_12_5x5_layer_call_and_return_conditional_losses_3614834�
'contract_12_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_12_5x5/StatefulPartitionedCall:output:0contract_12_3x3_3617543contract_12_3x3_3617545*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_12_3x3_layer_call_and_return_conditional_losses_3614851�
skip_12/PartitionedCallPartitionedCall0contract_12_3x3/StatefulPartitionedCall:output:0 skip_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_12_layer_call_and_return_conditional_losses_3614863�
concatenate_11/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_11_layer_call_and_return_conditional_losses_3614872�
%expand_13_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_11/PartitionedCall:output:0expand_13_5x5_3617550expand_13_5x5_3617552*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_13_5x5_layer_call_and_return_conditional_losses_3614885�
'contract_13_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_13_5x5/StatefulPartitionedCall:output:0contract_13_3x3_3617555contract_13_3x3_3617557*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_13_3x3_layer_call_and_return_conditional_losses_3614902�
skip_13/PartitionedCallPartitionedCall0contract_13_3x3/StatefulPartitionedCall:output:0 skip_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_13_layer_call_and_return_conditional_losses_3614914�
concatenate_12/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_12_layer_call_and_return_conditional_losses_3614923�
%expand_14_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_12/PartitionedCall:output:0expand_14_5x5_3617562expand_14_5x5_3617564*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_14_5x5_layer_call_and_return_conditional_losses_3614936�
'contract_14_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_14_5x5/StatefulPartitionedCall:output:0contract_14_3x3_3617567contract_14_3x3_3617569*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_14_3x3_layer_call_and_return_conditional_losses_3614953�
skip_14/PartitionedCallPartitionedCall0contract_14_3x3/StatefulPartitionedCall:output:0 skip_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_14_layer_call_and_return_conditional_losses_3614965�
concatenate_13/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_14/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_13_layer_call_and_return_conditional_losses_3614974�
%expand_15_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_13/PartitionedCall:output:0expand_15_5x5_3617574expand_15_5x5_3617576*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_15_5x5_layer_call_and_return_conditional_losses_3614987�
'contract_15_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_15_5x5/StatefulPartitionedCall:output:0contract_15_3x3_3617579contract_15_3x3_3617581*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_15_3x3_layer_call_and_return_conditional_losses_3615004�
skip_15/PartitionedCallPartitionedCall0contract_15_3x3/StatefulPartitionedCall:output:0 skip_14/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_15_layer_call_and_return_conditional_losses_3615016�
concatenate_14/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_14_layer_call_and_return_conditional_losses_3615025�
%expand_16_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_14/PartitionedCall:output:0expand_16_5x5_3617586expand_16_5x5_3617588*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_16_5x5_layer_call_and_return_conditional_losses_3615038�
'contract_16_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_16_5x5/StatefulPartitionedCall:output:0contract_16_3x3_3617591contract_16_3x3_3617593*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_16_3x3_layer_call_and_return_conditional_losses_3615055�
skip_16/PartitionedCallPartitionedCall0contract_16_3x3/StatefulPartitionedCall:output:0 skip_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_16_layer_call_and_return_conditional_losses_3615067�
concatenate_15/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_15_layer_call_and_return_conditional_losses_3615076�
%expand_17_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_15/PartitionedCall:output:0expand_17_5x5_3617598expand_17_5x5_3617600*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_17_5x5_layer_call_and_return_conditional_losses_3615089�
'contract_17_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_17_5x5/StatefulPartitionedCall:output:0contract_17_3x3_3617603contract_17_3x3_3617605*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_17_3x3_layer_call_and_return_conditional_losses_3615106�
skip_17/PartitionedCallPartitionedCall0contract_17_3x3/StatefulPartitionedCall:output:0 skip_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_17_layer_call_and_return_conditional_losses_3615118�
concatenate_16/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_16_layer_call_and_return_conditional_losses_3615127�
%expand_18_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_16/PartitionedCall:output:0expand_18_5x5_3617610expand_18_5x5_3617612*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_18_5x5_layer_call_and_return_conditional_losses_3615140�
'contract_18_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_18_5x5/StatefulPartitionedCall:output:0contract_18_3x3_3617615contract_18_3x3_3617617*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_18_3x3_layer_call_and_return_conditional_losses_3615157�
skip_18/PartitionedCallPartitionedCall0contract_18_3x3/StatefulPartitionedCall:output:0 skip_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_18_layer_call_and_return_conditional_losses_3615169�
concatenate_17/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_18/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_17_layer_call_and_return_conditional_losses_3615178�
%expand_19_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_17/PartitionedCall:output:0expand_19_5x5_3617622expand_19_5x5_3617624*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_19_5x5_layer_call_and_return_conditional_losses_3615191�
'contract_19_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_19_5x5/StatefulPartitionedCall:output:0contract_19_3x3_3617627contract_19_3x3_3617629*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_19_3x3_layer_call_and_return_conditional_losses_3615208�
skip_19/PartitionedCallPartitionedCall0contract_19_3x3/StatefulPartitionedCall:output:0 skip_18/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_19_layer_call_and_return_conditional_losses_3615220�
concatenate_18/PartitionedCallPartitionedCall3heuristic_priority/StatefulPartitionedCall:output:0 skip_19/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_18_layer_call_and_return_conditional_losses_3615229�
%expand_20_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_18/PartitionedCall:output:0expand_20_5x5_3617634expand_20_5x5_3617636*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_20_5x5_layer_call_and_return_conditional_losses_3615242�
'contract_20_3x3/StatefulPartitionedCallStatefulPartitionedCall.expand_20_5x5/StatefulPartitionedCall:output:0contract_20_3x3_3617639contract_20_3x3_3617641*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_20_3x3_layer_call_and_return_conditional_losses_3615259�
skip_20/PartitionedCallPartitionedCall0contract_20_3x3/StatefulPartitionedCall:output:0 skip_19/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_20_layer_call_and_return_conditional_losses_3615271�
all_value_input/PartitionedCallPartitionedCall0contract_20_3x3/StatefulPartitionedCall:output:0$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_all_value_input_layer_call_and_return_conditional_losses_3615280�
)policy_aggregator/StatefulPartitionedCallStatefulPartitionedCall skip_20/PartitionedCall:output:0policy_aggregator_3617646policy_aggregator_3617648*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_policy_aggregator_layer_call_and_return_conditional_losses_3615293�
 flat_value_input/PartitionedCallPartitionedCall(all_value_input/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������,* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_flat_value_input_layer_call_and_return_conditional_losses_3615305�
"border_off/StatefulPartitionedCallStatefulPartitionedCall2policy_aggregator/StatefulPartitionedCall:output:0border_off_3617652border_off_3617654*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_border_off_layer_call_and_return_conditional_losses_3615317^
tf.math.truediv/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �B�
tf.math.truediv/truedivRealDiv)flat_value_input/PartitionedCall:output:0"tf.math.truediv/truediv/y:output:0*
T0*(
_output_shapes
:����������,�
flat_logits/PartitionedCallPartitionedCall+border_off/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_flat_logits_layer_call_and_return_conditional_losses_3615331�
"value_head/StatefulPartitionedCallStatefulPartitionedCalltf.math.truediv/truediv:z:0value_head_3617660value_head_3617662*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_value_head_layer_call_and_return_conditional_losses_3615344�
policy_head/PartitionedCallPartitionedCall$flat_logits/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_policy_head_layer_call_and_return_conditional_losses_3615355t
IdentityIdentity$policy_head/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������|

Identity_1Identity+value_head/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp#^border_off/StatefulPartitionedCall(^contract_10_3x3/StatefulPartitionedCall(^contract_11_3x3/StatefulPartitionedCall(^contract_12_3x3/StatefulPartitionedCall(^contract_13_3x3/StatefulPartitionedCall(^contract_14_3x3/StatefulPartitionedCall(^contract_15_3x3/StatefulPartitionedCall(^contract_16_3x3/StatefulPartitionedCall(^contract_17_3x3/StatefulPartitionedCall(^contract_18_3x3/StatefulPartitionedCall(^contract_19_3x3/StatefulPartitionedCall'^contract_1_5x5/StatefulPartitionedCall(^contract_20_3x3/StatefulPartitionedCall'^contract_2_3x3/StatefulPartitionedCall'^contract_3_3x3/StatefulPartitionedCall'^contract_4_3x3/StatefulPartitionedCall'^contract_5_3x3/StatefulPartitionedCall'^contract_6_3x3/StatefulPartitionedCall'^contract_7_3x3/StatefulPartitionedCall'^contract_8_3x3/StatefulPartitionedCall'^contract_9_3x3/StatefulPartitionedCall&^expand_10_5x5/StatefulPartitionedCall&^expand_11_5x5/StatefulPartitionedCall&^expand_12_5x5/StatefulPartitionedCall&^expand_13_5x5/StatefulPartitionedCall&^expand_14_5x5/StatefulPartitionedCall&^expand_15_5x5/StatefulPartitionedCall&^expand_16_5x5/StatefulPartitionedCall&^expand_17_5x5/StatefulPartitionedCall&^expand_18_5x5/StatefulPartitionedCall&^expand_19_5x5/StatefulPartitionedCall'^expand_1_11x11/StatefulPartitionedCall&^expand_20_5x5/StatefulPartitionedCall%^expand_2_5x5/StatefulPartitionedCall%^expand_3_5x5/StatefulPartitionedCall%^expand_4_5x5/StatefulPartitionedCall%^expand_5_5x5/StatefulPartitionedCall%^expand_6_5x5/StatefulPartitionedCall%^expand_7_5x5/StatefulPartitionedCall%^expand_8_5x5/StatefulPartitionedCall%^expand_9_5x5/StatefulPartitionedCall+^heuristic_detector/StatefulPartitionedCall+^heuristic_priority/StatefulPartitionedCall*^policy_aggregator/StatefulPartitionedCall#^value_head/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
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
:���������
 
_user_specified_nameinputs
�
�
L__inference_contract_16_3x3_layer_call_and_return_conditional_losses_3618900

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
1__inference_contract_11_3x3_layer_call_fn_3618564

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_11_3x3_layer_call_and_return_conditional_losses_3614800w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
1__inference_contract_20_3x3_layer_call_fn_3619149

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_20_3x3_layer_call_and_return_conditional_losses_3615259w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
p
D__inference_skip_16_layer_call_and_return_conditional_losses_3618912
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
t
J__inference_concatenate_6_layer_call_and_return_conditional_losses_3614617

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
:���������	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
L__inference_contract_18_3x3_layer_call_and_return_conditional_losses_3615157

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
,__inference_border_off_layer_call_fn_3619214

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_border_off_layer_call_and_return_conditional_losses_3615317w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
��
�&
 __inference__traced_save_3619576
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

identity_1��MergeV2Checkpointsw
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
_temp/part�
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
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �)
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:]*
dtype0*�(
value�(B�(]B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-25/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-25/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-26/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-27/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-27/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-28/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-28/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-29/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-29/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-30/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-30/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-31/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-31/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-32/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-32/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-33/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-33/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-34/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-34/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-35/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-35/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-36/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-36/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-37/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-37/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-38/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-38/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-39/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-39/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-40/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-40/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-41/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-41/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-42/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-42/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-43/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-43/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-44/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-44/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:]*
dtype0*�
value�B�]B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �$
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_heuristic_detector_kernel_read_readvariableop2savev2_heuristic_detector_bias_read_readvariableop0savev2_expand_1_11x11_kernel_read_readvariableop.savev2_expand_1_11x11_bias_read_readvariableop4savev2_heuristic_priority_kernel_read_readvariableop2savev2_heuristic_priority_bias_read_readvariableop0savev2_contract_1_5x5_kernel_read_readvariableop.savev2_contract_1_5x5_bias_read_readvariableop.savev2_expand_2_5x5_kernel_read_readvariableop,savev2_expand_2_5x5_bias_read_readvariableop0savev2_contract_2_3x3_kernel_read_readvariableop.savev2_contract_2_3x3_bias_read_readvariableop.savev2_expand_3_5x5_kernel_read_readvariableop,savev2_expand_3_5x5_bias_read_readvariableop0savev2_contract_3_3x3_kernel_read_readvariableop.savev2_contract_3_3x3_bias_read_readvariableop.savev2_expand_4_5x5_kernel_read_readvariableop,savev2_expand_4_5x5_bias_read_readvariableop0savev2_contract_4_3x3_kernel_read_readvariableop.savev2_contract_4_3x3_bias_read_readvariableop.savev2_expand_5_5x5_kernel_read_readvariableop,savev2_expand_5_5x5_bias_read_readvariableop0savev2_contract_5_3x3_kernel_read_readvariableop.savev2_contract_5_3x3_bias_read_readvariableop.savev2_expand_6_5x5_kernel_read_readvariableop,savev2_expand_6_5x5_bias_read_readvariableop0savev2_contract_6_3x3_kernel_read_readvariableop.savev2_contract_6_3x3_bias_read_readvariableop.savev2_expand_7_5x5_kernel_read_readvariableop,savev2_expand_7_5x5_bias_read_readvariableop0savev2_contract_7_3x3_kernel_read_readvariableop.savev2_contract_7_3x3_bias_read_readvariableop.savev2_expand_8_5x5_kernel_read_readvariableop,savev2_expand_8_5x5_bias_read_readvariableop0savev2_contract_8_3x3_kernel_read_readvariableop.savev2_contract_8_3x3_bias_read_readvariableop.savev2_expand_9_5x5_kernel_read_readvariableop,savev2_expand_9_5x5_bias_read_readvariableop0savev2_contract_9_3x3_kernel_read_readvariableop.savev2_contract_9_3x3_bias_read_readvariableop/savev2_expand_10_5x5_kernel_read_readvariableop-savev2_expand_10_5x5_bias_read_readvariableop1savev2_contract_10_3x3_kernel_read_readvariableop/savev2_contract_10_3x3_bias_read_readvariableop/savev2_expand_11_5x5_kernel_read_readvariableop-savev2_expand_11_5x5_bias_read_readvariableop1savev2_contract_11_3x3_kernel_read_readvariableop/savev2_contract_11_3x3_bias_read_readvariableop/savev2_expand_12_5x5_kernel_read_readvariableop-savev2_expand_12_5x5_bias_read_readvariableop1savev2_contract_12_3x3_kernel_read_readvariableop/savev2_contract_12_3x3_bias_read_readvariableop/savev2_expand_13_5x5_kernel_read_readvariableop-savev2_expand_13_5x5_bias_read_readvariableop1savev2_contract_13_3x3_kernel_read_readvariableop/savev2_contract_13_3x3_bias_read_readvariableop/savev2_expand_14_5x5_kernel_read_readvariableop-savev2_expand_14_5x5_bias_read_readvariableop1savev2_contract_14_3x3_kernel_read_readvariableop/savev2_contract_14_3x3_bias_read_readvariableop/savev2_expand_15_5x5_kernel_read_readvariableop-savev2_expand_15_5x5_bias_read_readvariableop1savev2_contract_15_3x3_kernel_read_readvariableop/savev2_contract_15_3x3_bias_read_readvariableop/savev2_expand_16_5x5_kernel_read_readvariableop-savev2_expand_16_5x5_bias_read_readvariableop1savev2_contract_16_3x3_kernel_read_readvariableop/savev2_contract_16_3x3_bias_read_readvariableop/savev2_expand_17_5x5_kernel_read_readvariableop-savev2_expand_17_5x5_bias_read_readvariableop1savev2_contract_17_3x3_kernel_read_readvariableop/savev2_contract_17_3x3_bias_read_readvariableop/savev2_expand_18_5x5_kernel_read_readvariableop-savev2_expand_18_5x5_bias_read_readvariableop1savev2_contract_18_3x3_kernel_read_readvariableop/savev2_contract_18_3x3_bias_read_readvariableop/savev2_expand_19_5x5_kernel_read_readvariableop-savev2_expand_19_5x5_bias_read_readvariableop1savev2_contract_19_3x3_kernel_read_readvariableop/savev2_contract_19_3x3_bias_read_readvariableop/savev2_expand_20_5x5_kernel_read_readvariableop-savev2_expand_20_5x5_bias_read_readvariableop1savev2_contract_20_3x3_kernel_read_readvariableop/savev2_contract_20_3x3_bias_read_readvariableop3savev2_policy_aggregator_kernel_read_readvariableop1savev2_policy_aggregator_bias_read_readvariableop,savev2_border_off_kernel_read_readvariableop*savev2_border_off_bias_read_readvariableop,savev2_value_head_kernel_read_readvariableop*savev2_value_head_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *k
dtypesa
_2]�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :�:�:�:�:�::�::	 : : ::	 : : ::	 : : ::	 : : ::	 : : ::	 : : ::	 : : ::	 : : ::	 : : ::	 : : ::	 : : ::	 : : ::	 : : ::	 : : ::	 : : ::	 : : ::	 : : ::	 : : ::	 : : ::::::	�,:: : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
:�:!

_output_shapes	
:�:-)
'
_output_shapes
:�:!

_output_shapes	
:�:-)
'
_output_shapes
:�: 

_output_shapes
::-)
'
_output_shapes
:�: 
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
:	�,: Z
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
�
�
0__inference_contract_9_3x3_layer_call_fn_3618434

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_9_3x3_layer_call_and_return_conditional_losses_3614698w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
1__inference_contract_15_3x3_layer_call_fn_3618824

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_15_3x3_layer_call_and_return_conditional_losses_3615004w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
v
J__inference_concatenate_3_layer_call_and_return_conditional_losses_3618145
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
:���������	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
n
D__inference_skip_12_layer_call_and_return_conditional_losses_3614863

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
U
)__inference_skip_17_layer_call_fn_3618971
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_17_layer_call_and_return_conditional_losses_3615118h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
J__inference_expand_12_5x5_layer_call_and_return_conditional_losses_3614834

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:��������� m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
N
2__inference_flat_value_input_layer_call_fn_3619229

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������,* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_flat_value_input_layer_call_and_return_conditional_losses_3615305a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������,"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
L__inference_contract_17_3x3_layer_call_and_return_conditional_losses_3618965

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
p
D__inference_skip_13_layer_call_and_return_conditional_losses_3618717
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
J__inference_expand_18_5x5_layer_call_and_return_conditional_losses_3615140

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:��������� m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
\
0__inference_concatenate_18_layer_call_fn_3619113
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_18_layer_call_and_return_conditional_losses_3615229h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
n
D__inference_skip_20_layer_call_and_return_conditional_losses_3615271

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
t
J__inference_concatenate_5_layer_call_and_return_conditional_losses_3614566

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
:���������	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
K__inference_expand_1_11x11_layer_call_and_return_conditional_losses_3617897

inputs9
conv2d_readvariableop_resource:�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������a
SoftplusSoftplusBiasAdd:output:0*
T0*0
_output_shapes
:����������n
IdentityIdentitySoftplus:activations:0^NoOp*
T0*0
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
i
M__inference_flat_value_input_layer_call_and_return_conditional_losses_3615305

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����e  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������,Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������,"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
p
D__inference_skip_12_layer_call_and_return_conditional_losses_3618652
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
K__inference_contract_7_3x3_layer_call_and_return_conditional_losses_3618315

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
L__inference_contract_19_3x3_layer_call_and_return_conditional_losses_3619095

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
[
/__inference_concatenate_1_layer_call_fn_3618008
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_3614362h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
L__inference_contract_11_3x3_layer_call_and_return_conditional_losses_3614800

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
\
0__inference_concatenate_13_layer_call_fn_3618788
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_13_layer_call_and_return_conditional_losses_3614974h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
L__inference_contract_16_3x3_layer_call_and_return_conditional_losses_3615055

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
K__inference_contract_4_3x3_layer_call_and_return_conditional_losses_3614443

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
/__inference_expand_14_5x5_layer_call_fn_3618739

inputs!
unknown:	 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_expand_14_5x5_layer_call_and_return_conditional_losses_3614936w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
I__inference_expand_7_5x5_layer_call_and_return_conditional_losses_3618295

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:��������� m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
1__inference_contract_10_3x3_layer_call_fn_3618499

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_contract_10_3x3_layer_call_and_return_conditional_losses_3614749w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
\
0__inference_concatenate_15_layer_call_fn_3618918
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_15_layer_call_and_return_conditional_losses_3615076h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
n
D__inference_skip_19_layer_call_and_return_conditional_losses_3615220

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
O__inference_heuristic_priority_layer_call_and_return_conditional_losses_3614281

inputs9
conv2d_readvariableop_resource:�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������X
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:���������_
IdentityIdentityTanh:y:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
U
)__inference_skip_14_layer_call_fn_3618776
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_14_layer_call_and_return_conditional_losses_3614965h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
[
/__inference_concatenate_8_layer_call_fn_3618463
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_8_layer_call_and_return_conditional_losses_3614719h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
u
K__inference_concatenate_16_layer_call_and_return_conditional_losses_3615127

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
:���������	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
U
)__inference_skip_10_layer_call_fn_3618516
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_skip_10_layer_call_and_return_conditional_losses_3614761h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
m
C__inference_skip_4_layer_call_and_return_conditional_losses_3614455

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
m
C__inference_skip_9_layer_call_and_return_conditional_losses_3614710

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
K__inference_contract_5_3x3_layer_call_and_return_conditional_losses_3614494

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
K__inference_contract_8_3x3_layer_call_and_return_conditional_losses_3614647

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
K__inference_contract_8_3x3_layer_call_and_return_conditional_losses_3618380

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
.__inference_expand_3_5x5_layer_call_fn_3618024

inputs!
unknown:	 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_expand_3_5x5_layer_call_and_return_conditional_losses_3614375w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
J__inference_expand_10_5x5_layer_call_and_return_conditional_losses_3618490

inputs8
conv2d_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� `
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:��������� m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�

�
G__inference_value_head_layer_call_and_return_conditional_losses_3619276

inputs1
matmul_readvariableop_resource:	�,-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�,*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������,: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������,
 
_user_specified_nameinputs
�
�
0__inference_contract_4_3x3_layer_call_fn_3618109

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_4_3x3_layer_call_and_return_conditional_losses_3614443w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
n
D__inference_skip_17_layer_call_and_return_conditional_losses_3615118

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
L__inference_contract_13_3x3_layer_call_and_return_conditional_losses_3618705

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������`
SoftplusSoftplusBiasAdd:output:0*
T0*/
_output_shapes
:���������m
IdentityIdentitySoftplus:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
o
C__inference_skip_7_layer_call_and_return_conditional_losses_3618327
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
v
J__inference_concatenate_9_layer_call_and_return_conditional_losses_3618535
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
:���������	_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
[
/__inference_concatenate_4_layer_call_fn_3618203
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_4_layer_call_and_return_conditional_losses_3614515h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
0__inference_contract_5_3x3_layer_call_fn_3618174

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_contract_5_3x3_layer_call_and_return_conditional_losses_3614494w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
A
inputs7
serving_default_inputs:0���������@
policy_head1
StatefulPartitionedCall:0����������>

value_head0
StatefulPartitionedCall:1���������tensorflow/serving/predict:��
�
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
�
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
�
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
�
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
�
	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
j0
k1
s2
t3
|4
}5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
�58
�59
�60
�61
�62
�63
�64
�65
�66
�67
�68
�69
�70
�71
�72
�73
�74
�75
�76
�77
�78
�79
�80
�81
�82
�83
�84
�85
�86
�87
�88
�89"
trackable_list_wrapper
�
s0
t1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
�58
�59
�60
�61
�62
�63
�64
�65
�66
�67
�68
�69
�70
�71
�72
�73
�74
�75
�76
�77
�78
�79
�80
�81
�82
�83"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
`_default_save_signature
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
/__inference_gomoku_resnet_layer_call_fn_3615544
/__inference_gomoku_resnet_layer_call_fn_3617120�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
J__inference_gomoku_resnet_layer_call_and_return_conditional_losses_3617394
J__inference_gomoku_resnet_layer_call_and_return_conditional_losses_3617668�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z�trace_0z�trace_1
�B�
"__inference__wrapped_model_3614229inputs"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
	optimizer
 "
trackable_dict_wrapper
-
�serving_default"
signature_map
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
4__inference_heuristic_detector_layer_call_fn_3617866�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
O__inference_heuristic_detector_layer_call_and_return_conditional_losses_3617877�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
4:2�2heuristic_detector/kernel
&:$�2heuristic_detector/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_expand_1_11x11_layer_call_fn_3617886�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_expand_1_11x11_layer_call_and_return_conditional_losses_3617897�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0:.�2expand_1_11x11/kernel
": �2expand_1_11x11/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
4__inference_heuristic_priority_layer_call_fn_3617906�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
O__inference_heuristic_priority_layer_call_and_return_conditional_losses_3617917�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
4:2�2heuristic_priority/kernel
%:#2heuristic_priority/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_contract_1_5x5_layer_call_fn_3617926�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_contract_1_5x5_layer_call_and_return_conditional_losses_3617937�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0:.�2contract_1_5x5/kernel
!:2contract_1_5x5/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_concatenate_layer_call_fn_3617943�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
H__inference_concatenate_layer_call_and_return_conditional_losses_3617950�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_expand_2_5x5_layer_call_fn_3617959�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_expand_2_5x5_layer_call_and_return_conditional_losses_3617970�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
-:+	 2expand_2_5x5/kernel
: 2expand_2_5x5/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_contract_2_3x3_layer_call_fn_3617979�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_contract_2_3x3_layer_call_and_return_conditional_losses_3617990�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
/:- 2contract_2_3x3/kernel
!:2contract_2_3x3/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_skip_2_layer_call_fn_3617996�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_skip_2_layer_call_and_return_conditional_losses_3618002�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_concatenate_1_layer_call_fn_3618008�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
J__inference_concatenate_1_layer_call_and_return_conditional_losses_3618015�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_expand_3_5x5_layer_call_fn_3618024�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_expand_3_5x5_layer_call_and_return_conditional_losses_3618035�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
-:+	 2expand_3_5x5/kernel
: 2expand_3_5x5/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_contract_3_3x3_layer_call_fn_3618044�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_contract_3_3x3_layer_call_and_return_conditional_losses_3618055�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
/:- 2contract_3_3x3/kernel
!:2contract_3_3x3/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_skip_3_layer_call_fn_3618061�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_skip_3_layer_call_and_return_conditional_losses_3618067�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_concatenate_2_layer_call_fn_3618073�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
J__inference_concatenate_2_layer_call_and_return_conditional_losses_3618080�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_expand_4_5x5_layer_call_fn_3618089�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_expand_4_5x5_layer_call_and_return_conditional_losses_3618100�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
-:+	 2expand_4_5x5/kernel
: 2expand_4_5x5/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_contract_4_3x3_layer_call_fn_3618109�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_contract_4_3x3_layer_call_and_return_conditional_losses_3618120�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
/:- 2contract_4_3x3/kernel
!:2contract_4_3x3/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_skip_4_layer_call_fn_3618126�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_skip_4_layer_call_and_return_conditional_losses_3618132�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_concatenate_3_layer_call_fn_3618138�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
J__inference_concatenate_3_layer_call_and_return_conditional_losses_3618145�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_expand_5_5x5_layer_call_fn_3618154�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_expand_5_5x5_layer_call_and_return_conditional_losses_3618165�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
-:+	 2expand_5_5x5/kernel
: 2expand_5_5x5/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_contract_5_3x3_layer_call_fn_3618174�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_contract_5_3x3_layer_call_and_return_conditional_losses_3618185�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
/:- 2contract_5_3x3/kernel
!:2contract_5_3x3/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_skip_5_layer_call_fn_3618191�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_skip_5_layer_call_and_return_conditional_losses_3618197�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_concatenate_4_layer_call_fn_3618203�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
J__inference_concatenate_4_layer_call_and_return_conditional_losses_3618210�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_expand_6_5x5_layer_call_fn_3618219�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_expand_6_5x5_layer_call_and_return_conditional_losses_3618230�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
-:+	 2expand_6_5x5/kernel
: 2expand_6_5x5/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_contract_6_3x3_layer_call_fn_3618239�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_contract_6_3x3_layer_call_and_return_conditional_losses_3618250�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
/:- 2contract_6_3x3/kernel
!:2contract_6_3x3/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_skip_6_layer_call_fn_3618256�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_skip_6_layer_call_and_return_conditional_losses_3618262�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_concatenate_5_layer_call_fn_3618268�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
J__inference_concatenate_5_layer_call_and_return_conditional_losses_3618275�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_expand_7_5x5_layer_call_fn_3618284�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_expand_7_5x5_layer_call_and_return_conditional_losses_3618295�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
-:+	 2expand_7_5x5/kernel
: 2expand_7_5x5/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_contract_7_3x3_layer_call_fn_3618304�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_contract_7_3x3_layer_call_and_return_conditional_losses_3618315�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
/:- 2contract_7_3x3/kernel
!:2contract_7_3x3/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_skip_7_layer_call_fn_3618321�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_skip_7_layer_call_and_return_conditional_losses_3618327�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_concatenate_6_layer_call_fn_3618333�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
J__inference_concatenate_6_layer_call_and_return_conditional_losses_3618340�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_expand_8_5x5_layer_call_fn_3618349�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_expand_8_5x5_layer_call_and_return_conditional_losses_3618360�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
-:+	 2expand_8_5x5/kernel
: 2expand_8_5x5/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_contract_8_3x3_layer_call_fn_3618369�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_contract_8_3x3_layer_call_and_return_conditional_losses_3618380�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
/:- 2contract_8_3x3/kernel
!:2contract_8_3x3/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_skip_8_layer_call_fn_3618386�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_skip_8_layer_call_and_return_conditional_losses_3618392�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_concatenate_7_layer_call_fn_3618398�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
J__inference_concatenate_7_layer_call_and_return_conditional_losses_3618405�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_expand_9_5x5_layer_call_fn_3618414�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_expand_9_5x5_layer_call_and_return_conditional_losses_3618425�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
-:+	 2expand_9_5x5/kernel
: 2expand_9_5x5/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_contract_9_3x3_layer_call_fn_3618434�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_contract_9_3x3_layer_call_and_return_conditional_losses_3618445�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
/:- 2contract_9_3x3/kernel
!:2contract_9_3x3/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_skip_9_layer_call_fn_3618451�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_skip_9_layer_call_and_return_conditional_losses_3618457�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_concatenate_8_layer_call_fn_3618463�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
J__inference_concatenate_8_layer_call_and_return_conditional_losses_3618470�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_expand_10_5x5_layer_call_fn_3618479�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
J__inference_expand_10_5x5_layer_call_and_return_conditional_losses_3618490�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.:,	 2expand_10_5x5/kernel
 : 2expand_10_5x5/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_contract_10_3x3_layer_call_fn_3618499�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
L__inference_contract_10_3x3_layer_call_and_return_conditional_losses_3618510�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0:. 2contract_10_3x3/kernel
": 2contract_10_3x3/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_skip_10_layer_call_fn_3618516�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_skip_10_layer_call_and_return_conditional_losses_3618522�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_concatenate_9_layer_call_fn_3618528�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
J__inference_concatenate_9_layer_call_and_return_conditional_losses_3618535�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_expand_11_5x5_layer_call_fn_3618544�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
J__inference_expand_11_5x5_layer_call_and_return_conditional_losses_3618555�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.:,	 2expand_11_5x5/kernel
 : 2expand_11_5x5/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_contract_11_3x3_layer_call_fn_3618564�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
L__inference_contract_11_3x3_layer_call_and_return_conditional_losses_3618575�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0:. 2contract_11_3x3/kernel
": 2contract_11_3x3/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_skip_11_layer_call_fn_3618581�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_skip_11_layer_call_and_return_conditional_losses_3618587�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_concatenate_10_layer_call_fn_3618593�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_concatenate_10_layer_call_and_return_conditional_losses_3618600�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_expand_12_5x5_layer_call_fn_3618609�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
J__inference_expand_12_5x5_layer_call_and_return_conditional_losses_3618620�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.:,	 2expand_12_5x5/kernel
 : 2expand_12_5x5/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_contract_12_3x3_layer_call_fn_3618629�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
L__inference_contract_12_3x3_layer_call_and_return_conditional_losses_3618640�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0:. 2contract_12_3x3/kernel
": 2contract_12_3x3/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_skip_12_layer_call_fn_3618646�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_skip_12_layer_call_and_return_conditional_losses_3618652�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_concatenate_11_layer_call_fn_3618658�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_concatenate_11_layer_call_and_return_conditional_losses_3618665�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_expand_13_5x5_layer_call_fn_3618674�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
J__inference_expand_13_5x5_layer_call_and_return_conditional_losses_3618685�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.:,	 2expand_13_5x5/kernel
 : 2expand_13_5x5/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_contract_13_3x3_layer_call_fn_3618694�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
L__inference_contract_13_3x3_layer_call_and_return_conditional_losses_3618705�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0:. 2contract_13_3x3/kernel
": 2contract_13_3x3/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_skip_13_layer_call_fn_3618711�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_skip_13_layer_call_and_return_conditional_losses_3618717�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_concatenate_12_layer_call_fn_3618723�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_concatenate_12_layer_call_and_return_conditional_losses_3618730�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_expand_14_5x5_layer_call_fn_3618739�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
J__inference_expand_14_5x5_layer_call_and_return_conditional_losses_3618750�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.:,	 2expand_14_5x5/kernel
 : 2expand_14_5x5/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_contract_14_3x3_layer_call_fn_3618759�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
L__inference_contract_14_3x3_layer_call_and_return_conditional_losses_3618770�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0:. 2contract_14_3x3/kernel
": 2contract_14_3x3/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	non_trainable_variables
�	layers
�	metrics
 �	layer_regularization_losses
�	layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�	trace_02�
)__inference_skip_14_layer_call_fn_3618776�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	trace_0
�
�	trace_02�
D__inference_skip_14_layer_call_and_return_conditional_losses_3618782�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	non_trainable_variables
�	layers
�	metrics
 �	layer_regularization_losses
�	layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�	trace_02�
0__inference_concatenate_13_layer_call_fn_3618788�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	trace_0
�
�	trace_02�
K__inference_concatenate_13_layer_call_and_return_conditional_losses_3618795�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	non_trainable_variables
�	layers
�	metrics
 �	layer_regularization_losses
�	layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�	trace_02�
/__inference_expand_15_5x5_layer_call_fn_3618804�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	trace_0
�
�	trace_02�
J__inference_expand_15_5x5_layer_call_and_return_conditional_losses_3618815�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	trace_0
.:,	 2expand_15_5x5/kernel
 : 2expand_15_5x5/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	non_trainable_variables
�	layers
�	metrics
 �	layer_regularization_losses
�	layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�	trace_02�
1__inference_contract_15_3x3_layer_call_fn_3618824�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	trace_0
�
�	trace_02�
L__inference_contract_15_3x3_layer_call_and_return_conditional_losses_3618835�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	trace_0
0:. 2contract_15_3x3/kernel
": 2contract_15_3x3/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	non_trainable_variables
�	layers
�	metrics
 �	layer_regularization_losses
�	layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�	trace_02�
)__inference_skip_15_layer_call_fn_3618841�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	trace_0
�
�	trace_02�
D__inference_skip_15_layer_call_and_return_conditional_losses_3618847�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	non_trainable_variables
�	layers
�	metrics
 �	layer_regularization_losses
�	layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�	trace_02�
0__inference_concatenate_14_layer_call_fn_3618853�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	trace_0
�
�	trace_02�
K__inference_concatenate_14_layer_call_and_return_conditional_losses_3618860�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	non_trainable_variables
�	layers
�	metrics
 �	layer_regularization_losses
�	layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�	trace_02�
/__inference_expand_16_5x5_layer_call_fn_3618869�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	trace_0
�
�	trace_02�
J__inference_expand_16_5x5_layer_call_and_return_conditional_losses_3618880�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	trace_0
.:,	 2expand_16_5x5/kernel
 : 2expand_16_5x5/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	non_trainable_variables
�	layers
�	metrics
 �	layer_regularization_losses
�	layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�	trace_02�
1__inference_contract_16_3x3_layer_call_fn_3618889�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	trace_0
�
�	trace_02�
L__inference_contract_16_3x3_layer_call_and_return_conditional_losses_3618900�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	trace_0
0:. 2contract_16_3x3/kernel
": 2contract_16_3x3/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	non_trainable_variables
�	layers
�	metrics
 �	layer_regularization_losses
�	layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�	trace_02�
)__inference_skip_16_layer_call_fn_3618906�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	trace_0
�
�	trace_02�
D__inference_skip_16_layer_call_and_return_conditional_losses_3618912�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	non_trainable_variables
�	layers
�	metrics
 �	layer_regularization_losses
�	layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�	trace_02�
0__inference_concatenate_15_layer_call_fn_3618918�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	trace_0
�
�	trace_02�
K__inference_concatenate_15_layer_call_and_return_conditional_losses_3618925�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	non_trainable_variables
�	layers
�	metrics
 �	layer_regularization_losses
�	layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�	trace_02�
/__inference_expand_17_5x5_layer_call_fn_3618934�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	trace_0
�
�	trace_02�
J__inference_expand_17_5x5_layer_call_and_return_conditional_losses_3618945�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	trace_0
.:,	 2expand_17_5x5/kernel
 : 2expand_17_5x5/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	non_trainable_variables
�	layers
�	metrics
 �	layer_regularization_losses
�	layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�	trace_02�
1__inference_contract_17_3x3_layer_call_fn_3618954�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	trace_0
�
�	trace_02�
L__inference_contract_17_3x3_layer_call_and_return_conditional_losses_3618965�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	trace_0
0:. 2contract_17_3x3/kernel
": 2contract_17_3x3/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	non_trainable_variables
�	layers
�	metrics
 �	layer_regularization_losses
�	layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�	trace_02�
)__inference_skip_17_layer_call_fn_3618971�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	trace_0
�
�	trace_02�
D__inference_skip_17_layer_call_and_return_conditional_losses_3618977�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	non_trainable_variables
�	layers
�	metrics
 �	layer_regularization_losses
�	layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�	trace_02�
0__inference_concatenate_16_layer_call_fn_3618983�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	trace_0
�
�	trace_02�
K__inference_concatenate_16_layer_call_and_return_conditional_losses_3618990�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	non_trainable_variables
�	layers
�	metrics
 �	layer_regularization_losses
�	layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�	trace_02�
/__inference_expand_18_5x5_layer_call_fn_3618999�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	trace_0
�
�	trace_02�
J__inference_expand_18_5x5_layer_call_and_return_conditional_losses_3619010�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	trace_0
.:,	 2expand_18_5x5/kernel
 : 2expand_18_5x5/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	non_trainable_variables
�	layers
�	metrics
 �	layer_regularization_losses
�	layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�	trace_02�
1__inference_contract_18_3x3_layer_call_fn_3619019�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	trace_0
�
�	trace_02�
L__inference_contract_18_3x3_layer_call_and_return_conditional_losses_3619030�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	trace_0
0:. 2contract_18_3x3/kernel
": 2contract_18_3x3/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	non_trainable_variables
�	layers
�	metrics
 �	layer_regularization_losses
�	layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�	trace_02�
)__inference_skip_18_layer_call_fn_3619036�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	trace_0
�
�	trace_02�
D__inference_skip_18_layer_call_and_return_conditional_losses_3619042�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	non_trainable_variables
�	layers
�	metrics
 �	layer_regularization_losses
�	layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�	trace_02�
0__inference_concatenate_17_layer_call_fn_3619048�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	trace_0
�
�	trace_02�
K__inference_concatenate_17_layer_call_and_return_conditional_losses_3619055�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	non_trainable_variables
�	layers
�
metrics
 �
layer_regularization_losses
�
layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�
trace_02�
/__inference_expand_19_5x5_layer_call_fn_3619064�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�
trace_0
�
�
trace_02�
J__inference_expand_19_5x5_layer_call_and_return_conditional_losses_3619075�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�
trace_0
.:,	 2expand_19_5x5/kernel
 : 2expand_19_5x5/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�
non_trainable_variables
�
layers
�
metrics
 �
layer_regularization_losses
�
layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�
trace_02�
1__inference_contract_19_3x3_layer_call_fn_3619084�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�
trace_0
�
�
trace_02�
L__inference_contract_19_3x3_layer_call_and_return_conditional_losses_3619095�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�
trace_0
0:. 2contract_19_3x3/kernel
": 2contract_19_3x3/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�
non_trainable_variables
�
layers
�
metrics
 �
layer_regularization_losses
�
layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�
trace_02�
)__inference_skip_19_layer_call_fn_3619101�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�
trace_0
�
�
trace_02�
D__inference_skip_19_layer_call_and_return_conditional_losses_3619107�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�
trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�
non_trainable_variables
�
layers
�
metrics
 �
layer_regularization_losses
�
layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�
trace_02�
0__inference_concatenate_18_layer_call_fn_3619113�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�
trace_0
�
�
trace_02�
K__inference_concatenate_18_layer_call_and_return_conditional_losses_3619120�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�
trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�
non_trainable_variables
�
layers
�
metrics
 �
layer_regularization_losses
�
layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�
trace_02�
/__inference_expand_20_5x5_layer_call_fn_3619129�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�
trace_0
�
�
trace_02�
J__inference_expand_20_5x5_layer_call_and_return_conditional_losses_3619140�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�
trace_0
.:,	 2expand_20_5x5/kernel
 : 2expand_20_5x5/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�
non_trainable_variables
�
layers
�
metrics
 �
layer_regularization_losses
�
layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�
trace_02�
1__inference_contract_20_3x3_layer_call_fn_3619149�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�
trace_0
�
�
trace_02�
L__inference_contract_20_3x3_layer_call_and_return_conditional_losses_3619160�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�
trace_0
0:. 2contract_20_3x3/kernel
": 2contract_20_3x3/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�
non_trainable_variables
�
layers
�
metrics
 �
layer_regularization_losses
�
layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�
trace_02�
)__inference_skip_20_layer_call_fn_3619166�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�
trace_0
�
�
trace_02�
D__inference_skip_20_layer_call_and_return_conditional_losses_3619172�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�
trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�
non_trainable_variables
�
layers
�
metrics
 �
layer_regularization_losses
�
layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�
trace_02�
3__inference_policy_aggregator_layer_call_fn_3619181�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�
trace_0
�
�
trace_02�
N__inference_policy_aggregator_layer_call_and_return_conditional_losses_3619192�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�
trace_0
2:02policy_aggregator/kernel
$:"2policy_aggregator/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�
non_trainable_variables
�
layers
�
metrics
 �
layer_regularization_losses
�
layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�
trace_02�
1__inference_all_value_input_layer_call_fn_3619198�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�
trace_0
�
�
trace_02�
L__inference_all_value_input_layer_call_and_return_conditional_losses_3619205�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�
trace_0
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�
non_trainable_variables
�
layers
�
metrics
 �
layer_regularization_losses
�
layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�
trace_02�
,__inference_border_off_layer_call_fn_3619214�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�
trace_0
�
�
trace_02�
G__inference_border_off_layer_call_and_return_conditional_losses_3619224�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�
trace_0
+:)2border_off/kernel
:2border_off/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�
non_trainable_variables
�
layers
�
metrics
 �
layer_regularization_losses
�
layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�
trace_02�
2__inference_flat_value_input_layer_call_fn_3619229�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�
trace_0
�
�
trace_02�
M__inference_flat_value_input_layer_call_and_return_conditional_losses_3619235�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�
trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�
non_trainable_variables
�
layers
�
metrics
 �
layer_regularization_losses
�
layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�
trace_02�
-__inference_flat_logits_layer_call_fn_3619240�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�
trace_0
�
�
trace_02�
H__inference_flat_logits_layer_call_and_return_conditional_losses_3619246�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�
trace_0
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�
non_trainable_variables
�
layers
�
metrics
 �
layer_regularization_losses
�
layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�
trace_02�
-__inference_policy_head_layer_call_fn_3619251�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�
trace_0
�
�
trace_02�
H__inference_policy_head_layer_call_and_return_conditional_losses_3619256�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�
trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�
non_trainable_variables
�
layers
�
metrics
 �
layer_regularization_losses
�
layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�
trace_02�
,__inference_value_head_layer_call_fn_3619265�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�
trace_0
�
�
trace_02�
G__inference_value_head_layer_call_and_return_conditional_losses_3619276�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�
trace_0
$:"	�,2value_head/kernel
:2value_head/bias
L
j0
k1
|2
}3
�4
�5"
trackable_list_wrapper
�
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
�
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
/__inference_gomoku_resnet_layer_call_fn_3615544inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
/__inference_gomoku_resnet_layer_call_fn_3617120inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
J__inference_gomoku_resnet_layer_call_and_return_conditional_losses_3617394inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
J__inference_gomoku_resnet_layer_call_and_return_conditional_losses_3617668inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
%__inference_signature_wrapper_3617857inputs"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
4__inference_heuristic_detector_layer_call_fn_3617866inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_heuristic_detector_layer_call_and_return_conditional_losses_3617877inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
0__inference_expand_1_11x11_layer_call_fn_3617886inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_expand_1_11x11_layer_call_and_return_conditional_losses_3617897inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
4__inference_heuristic_priority_layer_call_fn_3617906inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_heuristic_priority_layer_call_and_return_conditional_losses_3617917inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
0__inference_contract_1_5x5_layer_call_fn_3617926inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_contract_1_5x5_layer_call_and_return_conditional_losses_3617937inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
-__inference_concatenate_layer_call_fn_3617943inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_concatenate_layer_call_and_return_conditional_losses_3617950inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
.__inference_expand_2_5x5_layer_call_fn_3617959inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_expand_2_5x5_layer_call_and_return_conditional_losses_3617970inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
0__inference_contract_2_3x3_layer_call_fn_3617979inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_contract_2_3x3_layer_call_and_return_conditional_losses_3617990inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
(__inference_skip_2_layer_call_fn_3617996inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_skip_2_layer_call_and_return_conditional_losses_3618002inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
/__inference_concatenate_1_layer_call_fn_3618008inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_concatenate_1_layer_call_and_return_conditional_losses_3618015inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
.__inference_expand_3_5x5_layer_call_fn_3618024inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_expand_3_5x5_layer_call_and_return_conditional_losses_3618035inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
0__inference_contract_3_3x3_layer_call_fn_3618044inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_contract_3_3x3_layer_call_and_return_conditional_losses_3618055inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
(__inference_skip_3_layer_call_fn_3618061inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_skip_3_layer_call_and_return_conditional_losses_3618067inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
/__inference_concatenate_2_layer_call_fn_3618073inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_concatenate_2_layer_call_and_return_conditional_losses_3618080inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
.__inference_expand_4_5x5_layer_call_fn_3618089inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_expand_4_5x5_layer_call_and_return_conditional_losses_3618100inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
0__inference_contract_4_3x3_layer_call_fn_3618109inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_contract_4_3x3_layer_call_and_return_conditional_losses_3618120inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
(__inference_skip_4_layer_call_fn_3618126inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_skip_4_layer_call_and_return_conditional_losses_3618132inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
/__inference_concatenate_3_layer_call_fn_3618138inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_concatenate_3_layer_call_and_return_conditional_losses_3618145inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
.__inference_expand_5_5x5_layer_call_fn_3618154inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_expand_5_5x5_layer_call_and_return_conditional_losses_3618165inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
0__inference_contract_5_3x3_layer_call_fn_3618174inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_contract_5_3x3_layer_call_and_return_conditional_losses_3618185inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
(__inference_skip_5_layer_call_fn_3618191inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_skip_5_layer_call_and_return_conditional_losses_3618197inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
/__inference_concatenate_4_layer_call_fn_3618203inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_concatenate_4_layer_call_and_return_conditional_losses_3618210inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
.__inference_expand_6_5x5_layer_call_fn_3618219inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_expand_6_5x5_layer_call_and_return_conditional_losses_3618230inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
0__inference_contract_6_3x3_layer_call_fn_3618239inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_contract_6_3x3_layer_call_and_return_conditional_losses_3618250inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
(__inference_skip_6_layer_call_fn_3618256inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_skip_6_layer_call_and_return_conditional_losses_3618262inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
/__inference_concatenate_5_layer_call_fn_3618268inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_concatenate_5_layer_call_and_return_conditional_losses_3618275inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
.__inference_expand_7_5x5_layer_call_fn_3618284inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_expand_7_5x5_layer_call_and_return_conditional_losses_3618295inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
0__inference_contract_7_3x3_layer_call_fn_3618304inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_contract_7_3x3_layer_call_and_return_conditional_losses_3618315inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
(__inference_skip_7_layer_call_fn_3618321inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_skip_7_layer_call_and_return_conditional_losses_3618327inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
/__inference_concatenate_6_layer_call_fn_3618333inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_concatenate_6_layer_call_and_return_conditional_losses_3618340inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
.__inference_expand_8_5x5_layer_call_fn_3618349inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_expand_8_5x5_layer_call_and_return_conditional_losses_3618360inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
0__inference_contract_8_3x3_layer_call_fn_3618369inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_contract_8_3x3_layer_call_and_return_conditional_losses_3618380inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
(__inference_skip_8_layer_call_fn_3618386inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_skip_8_layer_call_and_return_conditional_losses_3618392inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
/__inference_concatenate_7_layer_call_fn_3618398inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_concatenate_7_layer_call_and_return_conditional_losses_3618405inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
.__inference_expand_9_5x5_layer_call_fn_3618414inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_expand_9_5x5_layer_call_and_return_conditional_losses_3618425inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
0__inference_contract_9_3x3_layer_call_fn_3618434inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_contract_9_3x3_layer_call_and_return_conditional_losses_3618445inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
(__inference_skip_9_layer_call_fn_3618451inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_skip_9_layer_call_and_return_conditional_losses_3618457inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
/__inference_concatenate_8_layer_call_fn_3618463inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_concatenate_8_layer_call_and_return_conditional_losses_3618470inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
/__inference_expand_10_5x5_layer_call_fn_3618479inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_expand_10_5x5_layer_call_and_return_conditional_losses_3618490inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
1__inference_contract_10_3x3_layer_call_fn_3618499inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_contract_10_3x3_layer_call_and_return_conditional_losses_3618510inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
)__inference_skip_10_layer_call_fn_3618516inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_skip_10_layer_call_and_return_conditional_losses_3618522inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
/__inference_concatenate_9_layer_call_fn_3618528inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_concatenate_9_layer_call_and_return_conditional_losses_3618535inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
/__inference_expand_11_5x5_layer_call_fn_3618544inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_expand_11_5x5_layer_call_and_return_conditional_losses_3618555inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
1__inference_contract_11_3x3_layer_call_fn_3618564inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_contract_11_3x3_layer_call_and_return_conditional_losses_3618575inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
)__inference_skip_11_layer_call_fn_3618581inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_skip_11_layer_call_and_return_conditional_losses_3618587inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
0__inference_concatenate_10_layer_call_fn_3618593inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_concatenate_10_layer_call_and_return_conditional_losses_3618600inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
/__inference_expand_12_5x5_layer_call_fn_3618609inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_expand_12_5x5_layer_call_and_return_conditional_losses_3618620inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
1__inference_contract_12_3x3_layer_call_fn_3618629inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_contract_12_3x3_layer_call_and_return_conditional_losses_3618640inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
)__inference_skip_12_layer_call_fn_3618646inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_skip_12_layer_call_and_return_conditional_losses_3618652inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
0__inference_concatenate_11_layer_call_fn_3618658inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_concatenate_11_layer_call_and_return_conditional_losses_3618665inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
/__inference_expand_13_5x5_layer_call_fn_3618674inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_expand_13_5x5_layer_call_and_return_conditional_losses_3618685inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
1__inference_contract_13_3x3_layer_call_fn_3618694inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_contract_13_3x3_layer_call_and_return_conditional_losses_3618705inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
)__inference_skip_13_layer_call_fn_3618711inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_skip_13_layer_call_and_return_conditional_losses_3618717inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
0__inference_concatenate_12_layer_call_fn_3618723inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_concatenate_12_layer_call_and_return_conditional_losses_3618730inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
/__inference_expand_14_5x5_layer_call_fn_3618739inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_expand_14_5x5_layer_call_and_return_conditional_losses_3618750inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
1__inference_contract_14_3x3_layer_call_fn_3618759inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_contract_14_3x3_layer_call_and_return_conditional_losses_3618770inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
)__inference_skip_14_layer_call_fn_3618776inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_skip_14_layer_call_and_return_conditional_losses_3618782inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
0__inference_concatenate_13_layer_call_fn_3618788inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_concatenate_13_layer_call_and_return_conditional_losses_3618795inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
/__inference_expand_15_5x5_layer_call_fn_3618804inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_expand_15_5x5_layer_call_and_return_conditional_losses_3618815inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
1__inference_contract_15_3x3_layer_call_fn_3618824inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_contract_15_3x3_layer_call_and_return_conditional_losses_3618835inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
)__inference_skip_15_layer_call_fn_3618841inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_skip_15_layer_call_and_return_conditional_losses_3618847inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
0__inference_concatenate_14_layer_call_fn_3618853inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_concatenate_14_layer_call_and_return_conditional_losses_3618860inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
/__inference_expand_16_5x5_layer_call_fn_3618869inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_expand_16_5x5_layer_call_and_return_conditional_losses_3618880inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
1__inference_contract_16_3x3_layer_call_fn_3618889inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_contract_16_3x3_layer_call_and_return_conditional_losses_3618900inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
)__inference_skip_16_layer_call_fn_3618906inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_skip_16_layer_call_and_return_conditional_losses_3618912inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
0__inference_concatenate_15_layer_call_fn_3618918inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_concatenate_15_layer_call_and_return_conditional_losses_3618925inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
/__inference_expand_17_5x5_layer_call_fn_3618934inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_expand_17_5x5_layer_call_and_return_conditional_losses_3618945inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
1__inference_contract_17_3x3_layer_call_fn_3618954inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_contract_17_3x3_layer_call_and_return_conditional_losses_3618965inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
)__inference_skip_17_layer_call_fn_3618971inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_skip_17_layer_call_and_return_conditional_losses_3618977inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
0__inference_concatenate_16_layer_call_fn_3618983inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_concatenate_16_layer_call_and_return_conditional_losses_3618990inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
/__inference_expand_18_5x5_layer_call_fn_3618999inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_expand_18_5x5_layer_call_and_return_conditional_losses_3619010inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
1__inference_contract_18_3x3_layer_call_fn_3619019inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_contract_18_3x3_layer_call_and_return_conditional_losses_3619030inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
)__inference_skip_18_layer_call_fn_3619036inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_skip_18_layer_call_and_return_conditional_losses_3619042inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
0__inference_concatenate_17_layer_call_fn_3619048inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_concatenate_17_layer_call_and_return_conditional_losses_3619055inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
/__inference_expand_19_5x5_layer_call_fn_3619064inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_expand_19_5x5_layer_call_and_return_conditional_losses_3619075inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
1__inference_contract_19_3x3_layer_call_fn_3619084inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_contract_19_3x3_layer_call_and_return_conditional_losses_3619095inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
)__inference_skip_19_layer_call_fn_3619101inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_skip_19_layer_call_and_return_conditional_losses_3619107inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
0__inference_concatenate_18_layer_call_fn_3619113inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_concatenate_18_layer_call_and_return_conditional_losses_3619120inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
/__inference_expand_20_5x5_layer_call_fn_3619129inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_expand_20_5x5_layer_call_and_return_conditional_losses_3619140inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
1__inference_contract_20_3x3_layer_call_fn_3619149inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_contract_20_3x3_layer_call_and_return_conditional_losses_3619160inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
)__inference_skip_20_layer_call_fn_3619166inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_skip_20_layer_call_and_return_conditional_losses_3619172inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
3__inference_policy_aggregator_layer_call_fn_3619181inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_policy_aggregator_layer_call_and_return_conditional_losses_3619192inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
1__inference_all_value_input_layer_call_fn_3619198inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_all_value_input_layer_call_and_return_conditional_losses_3619205inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_border_off_layer_call_fn_3619214inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_border_off_layer_call_and_return_conditional_losses_3619224inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
2__inference_flat_value_input_layer_call_fn_3619229inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
M__inference_flat_value_input_layer_call_and_return_conditional_losses_3619235inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
-__inference_flat_logits_layer_call_fn_3619240inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_flat_logits_layer_call_and_return_conditional_losses_3619246inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
-__inference_policy_head_layer_call_fn_3619251inputs"�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_policy_head_layer_call_and_return_conditional_losses_3619256inputs"�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
,__inference_value_head_layer_call_fn_3619265inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_value_head_layer_call_and_return_conditional_losses_3619276inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�
	variables
�
	keras_api

�
total

�
count"
_tf_keras_metric
0
�
0
�
1"
trackable_list_wrapper
.
�
	variables"
_generic_user_object
:  (2total
:  (2count�
"__inference__wrapped_model_3614229��stjk|}������������������������������������������������������������������������������������7�4
-�*
(�%
inputs���������
� "n�k
5
policy_head&�#
policy_head����������
2

value_head$�!

value_head����������
L__inference_all_value_input_layer_call_and_return_conditional_losses_3619205�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������	
� "-�*
#� 
0���������
� �
1__inference_all_value_input_layer_call_fn_3619198�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������	
� " �����������
G__inference_border_off_layer_call_and_return_conditional_losses_3619224n��7�4
-�*
(�%
inputs���������
� "-�*
#� 
0���������
� �
,__inference_border_off_layer_call_fn_3619214a��7�4
-�*
(�%
inputs���������
� " �����������
K__inference_concatenate_10_layer_call_and_return_conditional_losses_3618600�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� "-�*
#� 
0���������	
� �
0__inference_concatenate_10_layer_call_fn_3618593�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� " ����������	�
K__inference_concatenate_11_layer_call_and_return_conditional_losses_3618665�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� "-�*
#� 
0���������	
� �
0__inference_concatenate_11_layer_call_fn_3618658�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� " ����������	�
K__inference_concatenate_12_layer_call_and_return_conditional_losses_3618730�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� "-�*
#� 
0���������	
� �
0__inference_concatenate_12_layer_call_fn_3618723�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� " ����������	�
K__inference_concatenate_13_layer_call_and_return_conditional_losses_3618795�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� "-�*
#� 
0���������	
� �
0__inference_concatenate_13_layer_call_fn_3618788�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� " ����������	�
K__inference_concatenate_14_layer_call_and_return_conditional_losses_3618860�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� "-�*
#� 
0���������	
� �
0__inference_concatenate_14_layer_call_fn_3618853�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� " ����������	�
K__inference_concatenate_15_layer_call_and_return_conditional_losses_3618925�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� "-�*
#� 
0���������	
� �
0__inference_concatenate_15_layer_call_fn_3618918�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� " ����������	�
K__inference_concatenate_16_layer_call_and_return_conditional_losses_3618990�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� "-�*
#� 
0���������	
� �
0__inference_concatenate_16_layer_call_fn_3618983�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� " ����������	�
K__inference_concatenate_17_layer_call_and_return_conditional_losses_3619055�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� "-�*
#� 
0���������	
� �
0__inference_concatenate_17_layer_call_fn_3619048�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� " ����������	�
K__inference_concatenate_18_layer_call_and_return_conditional_losses_3619120�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� "-�*
#� 
0���������	
� �
0__inference_concatenate_18_layer_call_fn_3619113�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� " ����������	�
J__inference_concatenate_1_layer_call_and_return_conditional_losses_3618015�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� "-�*
#� 
0���������	
� �
/__inference_concatenate_1_layer_call_fn_3618008�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� " ����������	�
J__inference_concatenate_2_layer_call_and_return_conditional_losses_3618080�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� "-�*
#� 
0���������	
� �
/__inference_concatenate_2_layer_call_fn_3618073�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� " ����������	�
J__inference_concatenate_3_layer_call_and_return_conditional_losses_3618145�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� "-�*
#� 
0���������	
� �
/__inference_concatenate_3_layer_call_fn_3618138�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� " ����������	�
J__inference_concatenate_4_layer_call_and_return_conditional_losses_3618210�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� "-�*
#� 
0���������	
� �
/__inference_concatenate_4_layer_call_fn_3618203�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� " ����������	�
J__inference_concatenate_5_layer_call_and_return_conditional_losses_3618275�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� "-�*
#� 
0���������	
� �
/__inference_concatenate_5_layer_call_fn_3618268�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� " ����������	�
J__inference_concatenate_6_layer_call_and_return_conditional_losses_3618340�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� "-�*
#� 
0���������	
� �
/__inference_concatenate_6_layer_call_fn_3618333�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� " ����������	�
J__inference_concatenate_7_layer_call_and_return_conditional_losses_3618405�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� "-�*
#� 
0���������	
� �
/__inference_concatenate_7_layer_call_fn_3618398�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� " ����������	�
J__inference_concatenate_8_layer_call_and_return_conditional_losses_3618470�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� "-�*
#� 
0���������	
� �
/__inference_concatenate_8_layer_call_fn_3618463�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� " ����������	�
J__inference_concatenate_9_layer_call_and_return_conditional_losses_3618535�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� "-�*
#� 
0���������	
� �
/__inference_concatenate_9_layer_call_fn_3618528�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� " ����������	�
H__inference_concatenate_layer_call_and_return_conditional_losses_3617950�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� "-�*
#� 
0���������	
� �
-__inference_concatenate_layer_call_fn_3617943�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� " ����������	�
L__inference_contract_10_3x3_layer_call_and_return_conditional_losses_3618510n��7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0���������
� �
1__inference_contract_10_3x3_layer_call_fn_3618499a��7�4
-�*
(�%
inputs��������� 
� " �����������
L__inference_contract_11_3x3_layer_call_and_return_conditional_losses_3618575n��7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0���������
� �
1__inference_contract_11_3x3_layer_call_fn_3618564a��7�4
-�*
(�%
inputs��������� 
� " �����������
L__inference_contract_12_3x3_layer_call_and_return_conditional_losses_3618640n��7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0���������
� �
1__inference_contract_12_3x3_layer_call_fn_3618629a��7�4
-�*
(�%
inputs��������� 
� " �����������
L__inference_contract_13_3x3_layer_call_and_return_conditional_losses_3618705n��7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0���������
� �
1__inference_contract_13_3x3_layer_call_fn_3618694a��7�4
-�*
(�%
inputs��������� 
� " �����������
L__inference_contract_14_3x3_layer_call_and_return_conditional_losses_3618770n��7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0���������
� �
1__inference_contract_14_3x3_layer_call_fn_3618759a��7�4
-�*
(�%
inputs��������� 
� " �����������
L__inference_contract_15_3x3_layer_call_and_return_conditional_losses_3618835n��7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0���������
� �
1__inference_contract_15_3x3_layer_call_fn_3618824a��7�4
-�*
(�%
inputs��������� 
� " �����������
L__inference_contract_16_3x3_layer_call_and_return_conditional_losses_3618900n��7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0���������
� �
1__inference_contract_16_3x3_layer_call_fn_3618889a��7�4
-�*
(�%
inputs��������� 
� " �����������
L__inference_contract_17_3x3_layer_call_and_return_conditional_losses_3618965n��7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0���������
� �
1__inference_contract_17_3x3_layer_call_fn_3618954a��7�4
-�*
(�%
inputs��������� 
� " �����������
L__inference_contract_18_3x3_layer_call_and_return_conditional_losses_3619030n��7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0���������
� �
1__inference_contract_18_3x3_layer_call_fn_3619019a��7�4
-�*
(�%
inputs��������� 
� " �����������
L__inference_contract_19_3x3_layer_call_and_return_conditional_losses_3619095n��7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0���������
� �
1__inference_contract_19_3x3_layer_call_fn_3619084a��7�4
-�*
(�%
inputs��������� 
� " �����������
K__inference_contract_1_5x5_layer_call_and_return_conditional_losses_3617937o��8�5
.�+
)�&
inputs����������
� "-�*
#� 
0���������
� �
0__inference_contract_1_5x5_layer_call_fn_3617926b��8�5
.�+
)�&
inputs����������
� " �����������
L__inference_contract_20_3x3_layer_call_and_return_conditional_losses_3619160n��7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0���������
� �
1__inference_contract_20_3x3_layer_call_fn_3619149a��7�4
-�*
(�%
inputs��������� 
� " �����������
K__inference_contract_2_3x3_layer_call_and_return_conditional_losses_3617990n��7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0���������
� �
0__inference_contract_2_3x3_layer_call_fn_3617979a��7�4
-�*
(�%
inputs��������� 
� " �����������
K__inference_contract_3_3x3_layer_call_and_return_conditional_losses_3618055n��7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0���������
� �
0__inference_contract_3_3x3_layer_call_fn_3618044a��7�4
-�*
(�%
inputs��������� 
� " �����������
K__inference_contract_4_3x3_layer_call_and_return_conditional_losses_3618120n��7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0���������
� �
0__inference_contract_4_3x3_layer_call_fn_3618109a��7�4
-�*
(�%
inputs��������� 
� " �����������
K__inference_contract_5_3x3_layer_call_and_return_conditional_losses_3618185n��7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0���������
� �
0__inference_contract_5_3x3_layer_call_fn_3618174a��7�4
-�*
(�%
inputs��������� 
� " �����������
K__inference_contract_6_3x3_layer_call_and_return_conditional_losses_3618250n��7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0���������
� �
0__inference_contract_6_3x3_layer_call_fn_3618239a��7�4
-�*
(�%
inputs��������� 
� " �����������
K__inference_contract_7_3x3_layer_call_and_return_conditional_losses_3618315n��7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0���������
� �
0__inference_contract_7_3x3_layer_call_fn_3618304a��7�4
-�*
(�%
inputs��������� 
� " �����������
K__inference_contract_8_3x3_layer_call_and_return_conditional_losses_3618380n��7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0���������
� �
0__inference_contract_8_3x3_layer_call_fn_3618369a��7�4
-�*
(�%
inputs��������� 
� " �����������
K__inference_contract_9_3x3_layer_call_and_return_conditional_losses_3618445n��7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0���������
� �
0__inference_contract_9_3x3_layer_call_fn_3618434a��7�4
-�*
(�%
inputs��������� 
� " �����������
J__inference_expand_10_5x5_layer_call_and_return_conditional_losses_3618490n��7�4
-�*
(�%
inputs���������	
� "-�*
#� 
0��������� 
� �
/__inference_expand_10_5x5_layer_call_fn_3618479a��7�4
-�*
(�%
inputs���������	
� " ���������� �
J__inference_expand_11_5x5_layer_call_and_return_conditional_losses_3618555n��7�4
-�*
(�%
inputs���������	
� "-�*
#� 
0��������� 
� �
/__inference_expand_11_5x5_layer_call_fn_3618544a��7�4
-�*
(�%
inputs���������	
� " ���������� �
J__inference_expand_12_5x5_layer_call_and_return_conditional_losses_3618620n��7�4
-�*
(�%
inputs���������	
� "-�*
#� 
0��������� 
� �
/__inference_expand_12_5x5_layer_call_fn_3618609a��7�4
-�*
(�%
inputs���������	
� " ���������� �
J__inference_expand_13_5x5_layer_call_and_return_conditional_losses_3618685n��7�4
-�*
(�%
inputs���������	
� "-�*
#� 
0��������� 
� �
/__inference_expand_13_5x5_layer_call_fn_3618674a��7�4
-�*
(�%
inputs���������	
� " ���������� �
J__inference_expand_14_5x5_layer_call_and_return_conditional_losses_3618750n��7�4
-�*
(�%
inputs���������	
� "-�*
#� 
0��������� 
� �
/__inference_expand_14_5x5_layer_call_fn_3618739a��7�4
-�*
(�%
inputs���������	
� " ���������� �
J__inference_expand_15_5x5_layer_call_and_return_conditional_losses_3618815n��7�4
-�*
(�%
inputs���������	
� "-�*
#� 
0��������� 
� �
/__inference_expand_15_5x5_layer_call_fn_3618804a��7�4
-�*
(�%
inputs���������	
� " ���������� �
J__inference_expand_16_5x5_layer_call_and_return_conditional_losses_3618880n��7�4
-�*
(�%
inputs���������	
� "-�*
#� 
0��������� 
� �
/__inference_expand_16_5x5_layer_call_fn_3618869a��7�4
-�*
(�%
inputs���������	
� " ���������� �
J__inference_expand_17_5x5_layer_call_and_return_conditional_losses_3618945n��7�4
-�*
(�%
inputs���������	
� "-�*
#� 
0��������� 
� �
/__inference_expand_17_5x5_layer_call_fn_3618934a��7�4
-�*
(�%
inputs���������	
� " ���������� �
J__inference_expand_18_5x5_layer_call_and_return_conditional_losses_3619010n��7�4
-�*
(�%
inputs���������	
� "-�*
#� 
0��������� 
� �
/__inference_expand_18_5x5_layer_call_fn_3618999a��7�4
-�*
(�%
inputs���������	
� " ���������� �
J__inference_expand_19_5x5_layer_call_and_return_conditional_losses_3619075n��7�4
-�*
(�%
inputs���������	
� "-�*
#� 
0��������� 
� �
/__inference_expand_19_5x5_layer_call_fn_3619064a��7�4
-�*
(�%
inputs���������	
� " ���������� �
K__inference_expand_1_11x11_layer_call_and_return_conditional_losses_3617897mst7�4
-�*
(�%
inputs���������
� ".�+
$�!
0����������
� �
0__inference_expand_1_11x11_layer_call_fn_3617886`st7�4
-�*
(�%
inputs���������
� "!������������
J__inference_expand_20_5x5_layer_call_and_return_conditional_losses_3619140n��7�4
-�*
(�%
inputs���������	
� "-�*
#� 
0��������� 
� �
/__inference_expand_20_5x5_layer_call_fn_3619129a��7�4
-�*
(�%
inputs���������	
� " ���������� �
I__inference_expand_2_5x5_layer_call_and_return_conditional_losses_3617970n��7�4
-�*
(�%
inputs���������	
� "-�*
#� 
0��������� 
� �
.__inference_expand_2_5x5_layer_call_fn_3617959a��7�4
-�*
(�%
inputs���������	
� " ���������� �
I__inference_expand_3_5x5_layer_call_and_return_conditional_losses_3618035n��7�4
-�*
(�%
inputs���������	
� "-�*
#� 
0��������� 
� �
.__inference_expand_3_5x5_layer_call_fn_3618024a��7�4
-�*
(�%
inputs���������	
� " ���������� �
I__inference_expand_4_5x5_layer_call_and_return_conditional_losses_3618100n��7�4
-�*
(�%
inputs���������	
� "-�*
#� 
0��������� 
� �
.__inference_expand_4_5x5_layer_call_fn_3618089a��7�4
-�*
(�%
inputs���������	
� " ���������� �
I__inference_expand_5_5x5_layer_call_and_return_conditional_losses_3618165n��7�4
-�*
(�%
inputs���������	
� "-�*
#� 
0��������� 
� �
.__inference_expand_5_5x5_layer_call_fn_3618154a��7�4
-�*
(�%
inputs���������	
� " ���������� �
I__inference_expand_6_5x5_layer_call_and_return_conditional_losses_3618230n��7�4
-�*
(�%
inputs���������	
� "-�*
#� 
0��������� 
� �
.__inference_expand_6_5x5_layer_call_fn_3618219a��7�4
-�*
(�%
inputs���������	
� " ���������� �
I__inference_expand_7_5x5_layer_call_and_return_conditional_losses_3618295n��7�4
-�*
(�%
inputs���������	
� "-�*
#� 
0��������� 
� �
.__inference_expand_7_5x5_layer_call_fn_3618284a��7�4
-�*
(�%
inputs���������	
� " ���������� �
I__inference_expand_8_5x5_layer_call_and_return_conditional_losses_3618360n��7�4
-�*
(�%
inputs���������	
� "-�*
#� 
0��������� 
� �
.__inference_expand_8_5x5_layer_call_fn_3618349a��7�4
-�*
(�%
inputs���������	
� " ���������� �
I__inference_expand_9_5x5_layer_call_and_return_conditional_losses_3618425n��7�4
-�*
(�%
inputs���������	
� "-�*
#� 
0��������� 
� �
.__inference_expand_9_5x5_layer_call_fn_3618414a��7�4
-�*
(�%
inputs���������	
� " ���������� �
H__inference_flat_logits_layer_call_and_return_conditional_losses_3619246a7�4
-�*
(�%
inputs���������
� "&�#
�
0����������
� �
-__inference_flat_logits_layer_call_fn_3619240T7�4
-�*
(�%
inputs���������
� "������������
M__inference_flat_value_input_layer_call_and_return_conditional_losses_3619235a7�4
-�*
(�%
inputs���������
� "&�#
�
0����������,
� �
2__inference_flat_value_input_layer_call_fn_3619229T7�4
-�*
(�%
inputs���������
� "�����������,�
J__inference_gomoku_resnet_layer_call_and_return_conditional_losses_3617394��stjk|}������������������������������������������������������������������������������������?�<
5�2
(�%
inputs���������
p 

 
� "L�I
B�?
�
0/0����������
�
0/1���������
� �
J__inference_gomoku_resnet_layer_call_and_return_conditional_losses_3617668��stjk|}������������������������������������������������������������������������������������?�<
5�2
(�%
inputs���������
p

 
� "L�I
B�?
�
0/0����������
�
0/1���������
� �
/__inference_gomoku_resnet_layer_call_fn_3615544��stjk|}������������������������������������������������������������������������������������?�<
5�2
(�%
inputs���������
p 

 
� ">�;
�
0����������
�
1����������
/__inference_gomoku_resnet_layer_call_fn_3617120��stjk|}������������������������������������������������������������������������������������?�<
5�2
(�%
inputs���������
p

 
� ">�;
�
0����������
�
1����������
O__inference_heuristic_detector_layer_call_and_return_conditional_losses_3617877mjk7�4
-�*
(�%
inputs���������
� ".�+
$�!
0����������
� �
4__inference_heuristic_detector_layer_call_fn_3617866`jk7�4
-�*
(�%
inputs���������
� "!������������
O__inference_heuristic_priority_layer_call_and_return_conditional_losses_3617917m|}8�5
.�+
)�&
inputs����������
� "-�*
#� 
0���������
� �
4__inference_heuristic_priority_layer_call_fn_3617906`|}8�5
.�+
)�&
inputs����������
� " �����������
N__inference_policy_aggregator_layer_call_and_return_conditional_losses_3619192n��7�4
-�*
(�%
inputs���������
� "-�*
#� 
0���������
� �
3__inference_policy_aggregator_layer_call_fn_3619181a��7�4
-�*
(�%
inputs���������
� " �����������
H__inference_policy_head_layer_call_and_return_conditional_losses_3619256^4�1
*�'
!�
inputs����������

 
� "&�#
�
0����������
� �
-__inference_policy_head_layer_call_fn_3619251Q4�1
*�'
!�
inputs����������

 
� "������������
%__inference_signature_wrapper_3617857��stjk|}������������������������������������������������������������������������������������A�>
� 
7�4
2
inputs(�%
inputs���������"n�k
5
policy_head&�#
policy_head����������
2

value_head$�!

value_head����������
D__inference_skip_10_layer_call_and_return_conditional_losses_3618522�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� "-�*
#� 
0���������
� �
)__inference_skip_10_layer_call_fn_3618516�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� " �����������
D__inference_skip_11_layer_call_and_return_conditional_losses_3618587�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� "-�*
#� 
0���������
� �
)__inference_skip_11_layer_call_fn_3618581�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� " �����������
D__inference_skip_12_layer_call_and_return_conditional_losses_3618652�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� "-�*
#� 
0���������
� �
)__inference_skip_12_layer_call_fn_3618646�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� " �����������
D__inference_skip_13_layer_call_and_return_conditional_losses_3618717�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� "-�*
#� 
0���������
� �
)__inference_skip_13_layer_call_fn_3618711�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� " �����������
D__inference_skip_14_layer_call_and_return_conditional_losses_3618782�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� "-�*
#� 
0���������
� �
)__inference_skip_14_layer_call_fn_3618776�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� " �����������
D__inference_skip_15_layer_call_and_return_conditional_losses_3618847�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� "-�*
#� 
0���������
� �
)__inference_skip_15_layer_call_fn_3618841�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� " �����������
D__inference_skip_16_layer_call_and_return_conditional_losses_3618912�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� "-�*
#� 
0���������
� �
)__inference_skip_16_layer_call_fn_3618906�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� " �����������
D__inference_skip_17_layer_call_and_return_conditional_losses_3618977�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� "-�*
#� 
0���������
� �
)__inference_skip_17_layer_call_fn_3618971�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� " �����������
D__inference_skip_18_layer_call_and_return_conditional_losses_3619042�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� "-�*
#� 
0���������
� �
)__inference_skip_18_layer_call_fn_3619036�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� " �����������
D__inference_skip_19_layer_call_and_return_conditional_losses_3619107�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� "-�*
#� 
0���������
� �
)__inference_skip_19_layer_call_fn_3619101�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� " �����������
D__inference_skip_20_layer_call_and_return_conditional_losses_3619172�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� "-�*
#� 
0���������
� �
)__inference_skip_20_layer_call_fn_3619166�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� " �����������
C__inference_skip_2_layer_call_and_return_conditional_losses_3618002�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� "-�*
#� 
0���������
� �
(__inference_skip_2_layer_call_fn_3617996�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� " �����������
C__inference_skip_3_layer_call_and_return_conditional_losses_3618067�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� "-�*
#� 
0���������
� �
(__inference_skip_3_layer_call_fn_3618061�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� " �����������
C__inference_skip_4_layer_call_and_return_conditional_losses_3618132�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� "-�*
#� 
0���������
� �
(__inference_skip_4_layer_call_fn_3618126�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� " �����������
C__inference_skip_5_layer_call_and_return_conditional_losses_3618197�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� "-�*
#� 
0���������
� �
(__inference_skip_5_layer_call_fn_3618191�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� " �����������
C__inference_skip_6_layer_call_and_return_conditional_losses_3618262�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� "-�*
#� 
0���������
� �
(__inference_skip_6_layer_call_fn_3618256�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� " �����������
C__inference_skip_7_layer_call_and_return_conditional_losses_3618327�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� "-�*
#� 
0���������
� �
(__inference_skip_7_layer_call_fn_3618321�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� " �����������
C__inference_skip_8_layer_call_and_return_conditional_losses_3618392�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� "-�*
#� 
0���������
� �
(__inference_skip_8_layer_call_fn_3618386�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� " �����������
C__inference_skip_9_layer_call_and_return_conditional_losses_3618457�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� "-�*
#� 
0���������
� �
(__inference_skip_9_layer_call_fn_3618451�j�g
`�]
[�X
*�'
inputs/0���������
*�'
inputs/1���������
� " �����������
G__inference_value_head_layer_call_and_return_conditional_losses_3619276_��0�-
&�#
!�
inputs����������,
� "%�"
�
0���������
� �
,__inference_value_head_layer_call_fn_3619265R��0�-
&�#
!�
inputs����������,
� "����������
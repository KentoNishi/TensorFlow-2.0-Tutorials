	
ß"Ť"
:
Add
x"T
y"T
z"T"
Ttype:
2	

ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeint
E
AssignAddVariableOp
resource
value"dtype"
dtypetype
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
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
š
DenseToDenseSetOperation	
set1"T	
set2"T
result_indices	
result_values"T
result_shape	"
set_operationstring"
validate_indicesbool("
Ttype:
	2	
5
DivNoNan
x"T
y"T
z"T"
Ttype:
2
B
Equal
x"T
y"T
z
"
Ttype:
2	

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
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

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
6
Pow
x"T
y"T
z"T"
Ttype:

2	
L
PreventGradient

input"T
output"T"	
Ttype"
messagestring 
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
ŕ
ResourceApplyAdam
var
m
v
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2

#SparseSoftmaxCrossEntropyWithLogits
features"T
labels"Tlabels	
loss"T
backprop"T"
Ttype:
2"
Tlabelstype0	:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape
9
VarIsInitializedOp
resource
is_initialized

&
	ZerosLike
x"T
y"T"	
Ttype"train*2.0.0-alpha02v1.12.0-9492-g2c319fb4158Â
s
dense_12_inputPlaceholder*
shape:˙˙˙˙˙˙˙˙˙*
dtype0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
0dense_12/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@dense_12/kernel*
valueB"     *
dtype0*
_output_shapes
:

.dense_12/kernel/Initializer/random_uniform/minConst*"
_class
loc:@dense_12/kernel*
valueB
 *HY˝*
dtype0*
_output_shapes
: 

.dense_12/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@dense_12/kernel*
valueB
 *HY=*
dtype0*
_output_shapes
: 
×
8dense_12/kernel/Initializer/random_uniform/RandomUniformRandomUniform0dense_12/kernel/Initializer/random_uniform/shape*
T0*"
_class
loc:@dense_12/kernel*
dtype0* 
_output_shapes
:

Ú
.dense_12/kernel/Initializer/random_uniform/subSub.dense_12/kernel/Initializer/random_uniform/max.dense_12/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_12/kernel*
_output_shapes
: 
î
.dense_12/kernel/Initializer/random_uniform/mulMul8dense_12/kernel/Initializer/random_uniform/RandomUniform.dense_12/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@dense_12/kernel* 
_output_shapes
:

ŕ
*dense_12/kernel/Initializer/random_uniformAdd.dense_12/kernel/Initializer/random_uniform/mul.dense_12/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_12/kernel* 
_output_shapes
:

 
dense_12/kernelVarHandleOp*
shape:
* 
shared_namedense_12/kernel*"
_class
loc:@dense_12/kernel*
dtype0*
_output_shapes
: 
o
0dense_12/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_12/kernel*
_output_shapes
: 

dense_12/kernel/AssignAssignVariableOpdense_12/kernel*dense_12/kernel/Initializer/random_uniform*"
_class
loc:@dense_12/kernel*
dtype0

#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*"
_class
loc:@dense_12/kernel*
dtype0* 
_output_shapes
:


dense_12/bias/Initializer/zerosConst* 
_class
loc:@dense_12/bias*
valueB*    *
dtype0*
_output_shapes	
:

dense_12/biasVarHandleOp*
shape:*
shared_namedense_12/bias* 
_class
loc:@dense_12/bias*
dtype0*
_output_shapes
: 
k
.dense_12/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_12/bias*
_output_shapes
: 

dense_12/bias/AssignAssignVariableOpdense_12/biasdense_12/bias/Initializer/zeros* 
_class
loc:@dense_12/bias*
dtype0

!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias* 
_class
loc:@dense_12/bias*
dtype0*
_output_shapes	
:
p
dense_12/MatMul/ReadVariableOpReadVariableOpdense_12/kernel*
dtype0* 
_output_shapes
:

|
dense_12/MatMulMatMuldense_12_inputdense_12/MatMul/ReadVariableOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
j
dense_12/BiasAdd/ReadVariableOpReadVariableOpdense_12/bias*
dtype0*
_output_shapes	
:

dense_12/BiasAddBiasAdddense_12/MatMuldense_12/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Z
dense_12/ReluReludense_12/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
[
dropout_6/dropout/rateConst*
valueB
 *ÍĚL>*
dtype0*
_output_shapes
: 
T
dropout_6/dropout/ShapeShapedense_12/Relu*
T0*
_output_shapes
:
i
$dropout_6/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
i
$dropout_6/dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

.dropout_6/dropout/random_uniform/RandomUniformRandomUniformdropout_6/dropout/Shape*
T0*
dtype0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

$dropout_6/dropout/random_uniform/subSub$dropout_6/dropout/random_uniform/max$dropout_6/dropout/random_uniform/min*
T0*
_output_shapes
: 
´
$dropout_6/dropout/random_uniform/mulMul.dropout_6/dropout/random_uniform/RandomUniform$dropout_6/dropout/random_uniform/sub*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ś
 dropout_6/dropout/random_uniformAdd$dropout_6/dropout/random_uniform/mul$dropout_6/dropout/random_uniform/min*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
dropout_6/dropout/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
n
dropout_6/dropout/subSubdropout_6/dropout/sub/xdropout_6/dropout/rate*
T0*
_output_shapes
: 
`
dropout_6/dropout/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
y
dropout_6/dropout/truedivRealDivdropout_6/dropout/truediv/xdropout_6/dropout/sub*
T0*
_output_shapes
: 

dropout_6/dropout/GreaterEqualGreaterEqual dropout_6/dropout/random_uniformdropout_6/dropout/rate*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
y
dropout_6/dropout/mulMuldense_12/Reludropout_6/dropout/truediv*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout_6/dropout/CastCastdropout_6/dropout/GreaterEqual*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0

dropout_6/dropout/mul_1Muldropout_6/dropout/muldropout_6/dropout/Cast*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
0dense_13/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@dense_13/kernel*
valueB"   
   *
dtype0*
_output_shapes
:

.dense_13/kernel/Initializer/random_uniform/minConst*"
_class
loc:@dense_13/kernel*
valueB
 *Ű˝*
dtype0*
_output_shapes
: 

.dense_13/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@dense_13/kernel*
valueB
 *Ű=*
dtype0*
_output_shapes
: 
Ö
8dense_13/kernel/Initializer/random_uniform/RandomUniformRandomUniform0dense_13/kernel/Initializer/random_uniform/shape*
T0*"
_class
loc:@dense_13/kernel*
dtype0*
_output_shapes
:	

Ú
.dense_13/kernel/Initializer/random_uniform/subSub.dense_13/kernel/Initializer/random_uniform/max.dense_13/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_13/kernel*
_output_shapes
: 
í
.dense_13/kernel/Initializer/random_uniform/mulMul8dense_13/kernel/Initializer/random_uniform/RandomUniform.dense_13/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@dense_13/kernel*
_output_shapes
:	

ß
*dense_13/kernel/Initializer/random_uniformAdd.dense_13/kernel/Initializer/random_uniform/mul.dense_13/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_13/kernel*
_output_shapes
:	


dense_13/kernelVarHandleOp*
shape:	
* 
shared_namedense_13/kernel*"
_class
loc:@dense_13/kernel*
dtype0*
_output_shapes
: 
o
0dense_13/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_13/kernel*
_output_shapes
: 

dense_13/kernel/AssignAssignVariableOpdense_13/kernel*dense_13/kernel/Initializer/random_uniform*"
_class
loc:@dense_13/kernel*
dtype0

#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*"
_class
loc:@dense_13/kernel*
dtype0*
_output_shapes
:	


dense_13/bias/Initializer/zerosConst* 
_class
loc:@dense_13/bias*
valueB
*    *
dtype0*
_output_shapes
:


dense_13/biasVarHandleOp*
shape:
*
shared_namedense_13/bias* 
_class
loc:@dense_13/bias*
dtype0*
_output_shapes
: 
k
.dense_13/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_13/bias*
_output_shapes
: 

dense_13/bias/AssignAssignVariableOpdense_13/biasdense_13/bias/Initializer/zeros* 
_class
loc:@dense_13/bias*
dtype0

!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias* 
_class
loc:@dense_13/bias*
dtype0*
_output_shapes
:

o
dense_13/MatMul/ReadVariableOpReadVariableOpdense_13/kernel*
dtype0*
_output_shapes
:	


dense_13/MatMulMatMuldropout_6/dropout/mul_1dense_13/MatMul/ReadVariableOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

i
dense_13/BiasAdd/ReadVariableOpReadVariableOpdense_13/bias*
dtype0*
_output_shapes
:


dense_13/BiasAddBiasAdddense_13/MatMuldense_13/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

_
dense_13/SoftmaxSoftmaxdense_13/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙


dense_13_targetPlaceholder*%
shape:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
dtype0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
R
ConstConst*
valueB*  ?*
dtype0*
_output_shapes
:

dense_13_sample_weightsPlaceholderWithDefaultConst*
shape:˙˙˙˙˙˙˙˙˙*
dtype0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
v
total/Initializer/zerosConst*
_class

loc:@total*
valueB
 *    *
dtype0*
_output_shapes
: 
x
totalVarHandleOp*
shape: *
shared_nametotal*
_class

loc:@total*
dtype0*
_output_shapes
: 
[
&total/IsInitialized/VarIsInitializedOpVarIsInitializedOptotal*
_output_shapes
: 
g
total/AssignAssignVariableOptotaltotal/Initializer/zeros*
_class

loc:@total*
dtype0
q
total/Read/ReadVariableOpReadVariableOptotal*
_class

loc:@total*
dtype0*
_output_shapes
: 
v
count/Initializer/zerosConst*
_class

loc:@count*
valueB
 *    *
dtype0*
_output_shapes
: 
x
countVarHandleOp*
shape: *
shared_namecount*
_class

loc:@count*
dtype0*
_output_shapes
: 
[
&count/IsInitialized/VarIsInitializedOpVarIsInitializedOpcount*
_output_shapes
: 
g
count/AssignAssignVariableOpcountcount/Initializer/zeros*
_class

loc:@count*
dtype0
q
count/Read/ReadVariableOpReadVariableOpcount*
_class

loc:@count*
dtype0*
_output_shapes
: 

metrics/accuracy/SqueezeSqueezedense_13_target*
squeeze_dims

˙˙˙˙˙˙˙˙˙*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
l
!metrics/accuracy/ArgMax/dimensionConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 

metrics/accuracy/ArgMaxArgMaxdense_13/Softmax!metrics/accuracy/ArgMax/dimension*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
metrics/accuracy/CastCastmetrics/accuracy/ArgMax*

SrcT0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0
~
metrics/accuracy/EqualEqualmetrics/accuracy/Squeezemetrics/accuracy/Cast*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
metrics/accuracy/Cast_1Castmetrics/accuracy/Equal*

SrcT0
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0
`
metrics/accuracy/ConstConst*
valueB: *
dtype0*
_output_shapes
:
m
metrics/accuracy/SumSummetrics/accuracy/Cast_1metrics/accuracy/Const*
T0*
_output_shapes
: 
e
$metrics/accuracy/AssignAddVariableOpAssignAddVariableOptotalmetrics/accuracy/Sum*
dtype0

metrics/accuracy/ReadVariableOpReadVariableOptotal%^metrics/accuracy/AssignAddVariableOp^metrics/accuracy/Sum*
dtype0*
_output_shapes
: 
W
metrics/accuracy/SizeSizemetrics/accuracy/Cast_1*
T0*
_output_shapes
: 
f
metrics/accuracy/Cast_2Castmetrics/accuracy/Size*

SrcT0*
_output_shapes
: *

DstT0

&metrics/accuracy/AssignAddVariableOp_1AssignAddVariableOpcountmetrics/accuracy/Cast_2%^metrics/accuracy/AssignAddVariableOp*
dtype0
Ż
!metrics/accuracy/ReadVariableOp_1ReadVariableOpcount%^metrics/accuracy/AssignAddVariableOp'^metrics/accuracy/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 

*metrics/accuracy/div_no_nan/ReadVariableOpReadVariableOptotal'^metrics/accuracy/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 

,metrics/accuracy/div_no_nan/ReadVariableOp_1ReadVariableOpcount'^metrics/accuracy/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
˘
metrics/accuracy/div_no_nanDivNoNan*metrics/accuracy/div_no_nan/ReadVariableOp,metrics/accuracy/div_no_nan/ReadVariableOp_1*
T0*
_output_shapes
: 
c
metrics/accuracy/IdentityIdentitymetrics/accuracy/div_no_nan*
T0*
_output_shapes
: 

metrics/accuracy/Squeeze_1Squeezedense_13_target*
squeeze_dims

˙˙˙˙˙˙˙˙˙*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
n
#metrics/accuracy/ArgMax_1/dimensionConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 

metrics/accuracy/ArgMax_1ArgMaxdense_13/Softmax#metrics/accuracy/ArgMax_1/dimension*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
w
metrics/accuracy/Cast_3Castmetrics/accuracy/ArgMax_1*

SrcT0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0

metrics/accuracy/Equal_1Equalmetrics/accuracy/Squeeze_1metrics/accuracy/Cast_3*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
v
metrics/accuracy/Cast_4Castmetrics/accuracy/Equal_1*

SrcT0
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0
b
metrics/accuracy/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
q
metrics/accuracy/MeanMeanmetrics/accuracy/Cast_4metrics/accuracy/Const_1*
T0*
_output_shapes
: 

@loss/dense_13_loss/sparse_categorical_crossentropy/Reshape/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
Ć
:loss/dense_13_loss/sparse_categorical_crossentropy/ReshapeReshapedense_13_target@loss/dense_13_loss/sparse_categorical_crossentropy/Reshape/shape*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
¸
7loss/dense_13_loss/sparse_categorical_crossentropy/CastCast:loss/dense_13_loss/sparse_categorical_crossentropy/Reshape*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0	

Bloss/dense_13_loss/sparse_categorical_crossentropy/Reshape_1/shapeConst*
valueB"˙˙˙˙
   *
dtype0*
_output_shapes
:
Ď
<loss/dense_13_loss/sparse_categorical_crossentropy/Reshape_1Reshapedense_13/BiasAddBloss/dense_13_loss/sparse_categorical_crossentropy/Reshape_1/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ă
\loss/dense_13_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/ShapeShape7loss/dense_13_loss/sparse_categorical_crossentropy/Cast*
T0	*
_output_shapes
:
Ů
zloss/dense_13_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits<loss/dense_13_loss/sparse_categorical_crossentropy/Reshape_17loss/dense_13_loss/sparse_categorical_crossentropy/Cast*
T0*6
_output_shapes$
":˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

ź
uloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/weights/shapeShapedense_13_sample_weights*
T0*
_output_shapes
:
ś
tloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B :*
dtype0*
_output_shapes
: 

tloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/values/shapeShapezloss/dense_13_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*
_output_shapes
:
ľ
sloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
ľ
sloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_scalar/xConst*
value	B : *
dtype0*
_output_shapes
: 

qloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_scalarEqualsloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_scalar/xtloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/weights/rank*
T0*
_output_shapes
: 

}loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/SwitchSwitchqloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_scalarqloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
­
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_tIdentityloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch:1*
T0
*
_output_shapes
: 
Ť
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_fIdentity}loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch*
T0
*
_output_shapes
: 

~loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_idIdentityqloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: 
Ś
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1Switchqloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_scalar~loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0
*
_classz
xvloc:@loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_scalar*
_output_shapes
: : 

loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqual¤loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchŚloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
T0*
_output_shapes
: 
Đ
¤loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitchsloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/values/rank~loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0*
_class|
zxloc:@loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/values/rank*
_output_shapes
: : 
Ô
Śloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1Switchtloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/weights/rank~loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0*
_class}
{yloc:@loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/weights/rank*
_output_shapes
: : 

loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitchloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: : 
ă
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentityloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
T0
*
_output_shapes
: 
á
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentityloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
T0
*
_output_shapes
: 
ć
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentityloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: 

°loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConst^loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
Ń
Źloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDimsˇloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1°loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*
T0*
_output_shapes

:
é
łloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitchtloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/values/shape~loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0*
_class}
{yloc:@loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/values/shape* 
_output_shapes
::
Ć
ľloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1Switchłloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switchloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*
_class}
{yloc:@loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/values/shape* 
_output_shapes
::
 
ąloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConst^loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB"      *
dtype0*
_output_shapes
:

ąloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConst^loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
Ĺ
Ťloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFilląloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shapeąloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
T0*
_output_shapes

:

­loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConst^loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
ô
¨loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2Źloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDimsŤloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like­loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*
T0*
N*
_output_shapes

:

˛loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConst^loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
×
Žloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDimsšloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1˛loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*
T0*
_output_shapes

:
í
ľloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitchuloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/weights/shape~loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0*
_class~
|zloc:@loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/weights/shape* 
_output_shapes
::
Ë
ˇloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1Switchľloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switchloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*
_class~
|zloc:@loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/weights/shape* 
_output_shapes
::

şloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperationŽloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1¨loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*
set_operationa-b*
T0*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:

˛loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSizeźloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
T0*
_output_shapes
: 

Łloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConst^loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B : *
dtype0*
_output_shapes
: 
§
Ąloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqualŁloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x˛loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
T0*
_output_shapes
: 
¸
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Switchloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0
*ł
_class¨
Ľ˘loc:@loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: : 

loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMergeloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Ąloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
T0
*
N*
_output_shapes
: : 
Î
|loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/MergeMergeloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Mergeloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:1*
T0
*
N*
_output_shapes
: : 
Ő
mloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/ConstConst*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
ž
oloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/Const_1Const*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
É
oloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/Const_2Const**
value!B Bdense_13_sample_weights:0*
dtype0*
_output_shapes
: 
˝
oloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/Const_3Const*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
Ż
oloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/Const_4Const*
valueB B|loss/dense_13_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:0*
dtype0*
_output_shapes
: 
ş
oloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/Const_5Const*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
Ł
zloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/SwitchSwitch|loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge|loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
§
|loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_tIdentity|loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
Ľ
|loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_fIdentityzloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Switch*
T0
*
_output_shapes
: 
Ś
{loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_idIdentity|loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: 
˙
xloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/NoOpNoOp}^loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_t
Â
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependencyIdentity|loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_ty^loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/NoOp*
T0
*
_class
loc:@loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_t*
_output_shapes
: 
é
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0Const}^loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
Đ
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1Const}^loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
Ű
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2Const}^loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f**
value!B Bdense_13_sample_weights:0*
dtype0*
_output_shapes
: 
Ď
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4Const}^loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
Á
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5Const}^loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
valueB B|loss/dense_13_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:0*
dtype0*
_output_shapes
: 
Ě
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7Const}^loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
Ć
zloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/AssertAssertloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switchloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3*
T
2	

ż
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/SwitchSwitch|loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge{loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*
T0
*
_class
loc:@loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: : 
¸
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1Switchuloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/weights/shape{loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*
T0*
_class~
|zloc:@loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/weights/shape* 
_output_shapes
::
ś
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2Switchtloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/values/shape{loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*
T0*
_class}
{yloc:@loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/values/shape* 
_output_shapes
::
¨
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3Switchqloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_scalar{loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*
T0
*
_classz
xvloc:@loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_scalar*
_output_shapes
: : 
Ć
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1Identity|loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f{^loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert*
T0
*
_class
loc:@loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: 
Â
yloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/MergeMergeloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 

bloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/ones_like/ShapeShapezloss/dense_13_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsz^loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Merge*
T0*
_output_shapes
:
Ł
bloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/ones_like/ConstConstz^loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Merge*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ú
\loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/ones_likeFillbloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/ones_like/Shapebloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/ones_like/Const*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ţ
Rloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weightsMuldense_13_sample_weights\loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/ones_like*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
É
Dloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/MulMulzloss/dense_13_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsRloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
loss/dense_13_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:

loss/dense_13_loss/SumSumDloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/Mulloss/dense_13_loss/Const*
T0*
_output_shapes
: 

loss/dense_13_loss/num_elementsSizeDloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/Mul*
T0*
_output_shapes
: 
}
$loss/dense_13_loss/num_elements/CastCastloss/dense_13_loss/num_elements*

SrcT0*
_output_shapes
: *

DstT0
]
loss/dense_13_loss/mul/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
~
loss/dense_13_loss/mulMulloss/dense_13_loss/mul/x$loss/dense_13_loss/num_elements/Cast*
T0*
_output_shapes
: 
]
loss/dense_13_loss/Const_1Const*
valueB *
dtype0*
_output_shapes
: 
t
loss/dense_13_loss/Sum_1Sumloss/dense_13_loss/Sumloss/dense_13_loss/Const_1*
T0*
_output_shapes
: 
w
loss/dense_13_loss/valueDivNoNanloss/dense_13_loss/Sum_1loss/dense_13_loss/mul*
T0*
_output_shapes
: 
O

loss/mul/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
V
loss/mulMul
loss/mul/xloss/dense_13_loss/value*
T0*
_output_shapes
: 
q
iter/Initializer/zerosConst*
_class
	loc:@iter*
value	B	 R *
dtype0	*
_output_shapes
: 

iterVarHandleOp"/device:CPU:0*
shape: *
shared_nameiter*
_class
	loc:@iter*
dtype0	*
_output_shapes
: 
h
%iter/IsInitialized/VarIsInitializedOpVarIsInitializedOpiter"/device:CPU:0*
_output_shapes
: 
r
iter/AssignAssignVariableOpiteriter/Initializer/zeros"/device:CPU:0*
_class
	loc:@iter*
dtype0	
}
iter/Read/ReadVariableOpReadVariableOpiter"/device:CPU:0*
_class
	loc:@iter*
dtype0	*
_output_shapes
: 

'learning_rate/Initializer/initial_valueConst* 
_class
loc:@learning_rate*
valueB
 *o:*
dtype0*
_output_shapes
: 

learning_rateVarHandleOp*
shape: *
shared_namelearning_rate* 
_class
loc:@learning_rate*
dtype0*
_output_shapes
: 
k
.learning_rate/IsInitialized/VarIsInitializedOpVarIsInitializedOplearning_rate*
_output_shapes
: 

learning_rate/AssignAssignVariableOplearning_rate'learning_rate/Initializer/initial_value* 
_class
loc:@learning_rate*
dtype0

!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate* 
_class
loc:@learning_rate*
dtype0*
_output_shapes
: 
~
decay/Initializer/initial_valueConst*
_class

loc:@decay*
valueB
 *    *
dtype0*
_output_shapes
: 
x
decayVarHandleOp*
shape: *
shared_namedecay*
_class

loc:@decay*
dtype0*
_output_shapes
: 
[
&decay/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecay*
_output_shapes
: 
o
decay/AssignAssignVariableOpdecaydecay/Initializer/initial_value*
_class

loc:@decay*
dtype0
q
decay/Read/ReadVariableOpReadVariableOpdecay*
_class

loc:@decay*
dtype0*
_output_shapes
: 

 beta_1/Initializer/initial_valueConst*
_class
loc:@beta_1*
valueB
 *fff?*
dtype0*
_output_shapes
: 
{
beta_1VarHandleOp*
shape: *
shared_namebeta_1*
_class
loc:@beta_1*
dtype0*
_output_shapes
: 
]
'beta_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpbeta_1*
_output_shapes
: 
s
beta_1/AssignAssignVariableOpbeta_1 beta_1/Initializer/initial_value*
_class
loc:@beta_1*
dtype0
t
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_class
loc:@beta_1*
dtype0*
_output_shapes
: 

 beta_2/Initializer/initial_valueConst*
_class
loc:@beta_2*
valueB
 *wž?*
dtype0*
_output_shapes
: 
{
beta_2VarHandleOp*
shape: *
shared_namebeta_2*
_class
loc:@beta_2*
dtype0*
_output_shapes
: 
]
'beta_2/IsInitialized/VarIsInitializedOpVarIsInitializedOpbeta_2*
_output_shapes
: 
s
beta_2/AssignAssignVariableOpbeta_2 beta_2/Initializer/initial_value*
_class
loc:@beta_2*
dtype0
t
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_class
loc:@beta_2*
dtype0*
_output_shapes
: 

!epsilon/Initializer/initial_valueConst*
_class
loc:@epsilon*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
~
epsilonVarHandleOp*
shape: *
shared_name	epsilon*
_class
loc:@epsilon*
dtype0*
_output_shapes
: 
_
(epsilon/IsInitialized/VarIsInitializedOpVarIsInitializedOpepsilon*
_output_shapes
: 
w
epsilon/AssignAssignVariableOpepsilon!epsilon/Initializer/initial_value*
_class
loc:@epsilon*
dtype0
w
epsilon/Read/ReadVariableOpReadVariableOpepsilon*
_class
loc:@epsilon*
dtype0*
_output_shapes
: 
`
training/Adam/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
f
!training/Adam/gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

training/Adam/gradients/FillFilltraining/Adam/gradients/Shape!training/Adam/gradients/grad_ys_0*
T0*
_output_shapes
: 

)training/Adam/gradients/loss/mul_grad/MulMultraining/Adam/gradients/Fillloss/dense_13_loss/value*
T0*
_output_shapes
: 
}
+training/Adam/gradients/loss/mul_grad/Mul_1Multraining/Adam/gradients/Fill
loss/mul/x*
T0*
_output_shapes
: 
~
;training/Adam/gradients/loss/dense_13_loss/value_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 

=training/Adam/gradients/loss/dense_13_loss/value_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 

Ktraining/Adam/gradients/loss/dense_13_loss/value_grad/BroadcastGradientArgsBroadcastGradientArgs;training/Adam/gradients/loss/dense_13_loss/value_grad/Shape=training/Adam/gradients/loss/dense_13_loss/value_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
˛
@training/Adam/gradients/loss/dense_13_loss/value_grad/div_no_nanDivNoNan+training/Adam/gradients/loss/mul_grad/Mul_1loss/dense_13_loss/mul*
T0*
_output_shapes
: 
đ
9training/Adam/gradients/loss/dense_13_loss/value_grad/SumSum@training/Adam/gradients/loss/dense_13_loss/value_grad/div_no_nanKtraining/Adam/gradients/loss/dense_13_loss/value_grad/BroadcastGradientArgs*
T0*
_output_shapes
: 
á
=training/Adam/gradients/loss/dense_13_loss/value_grad/ReshapeReshape9training/Adam/gradients/loss/dense_13_loss/value_grad/Sum;training/Adam/gradients/loss/dense_13_loss/value_grad/Shape*
T0*
_output_shapes
: 
{
9training/Adam/gradients/loss/dense_13_loss/value_grad/NegNegloss/dense_13_loss/Sum_1*
T0*
_output_shapes
: 
Â
Btraining/Adam/gradients/loss/dense_13_loss/value_grad/div_no_nan_1DivNoNan9training/Adam/gradients/loss/dense_13_loss/value_grad/Negloss/dense_13_loss/mul*
T0*
_output_shapes
: 
Ë
Btraining/Adam/gradients/loss/dense_13_loss/value_grad/div_no_nan_2DivNoNanBtraining/Adam/gradients/loss/dense_13_loss/value_grad/div_no_nan_1loss/dense_13_loss/mul*
T0*
_output_shapes
: 
Ň
9training/Adam/gradients/loss/dense_13_loss/value_grad/mulMul+training/Adam/gradients/loss/mul_grad/Mul_1Btraining/Adam/gradients/loss/dense_13_loss/value_grad/div_no_nan_2*
T0*
_output_shapes
: 
í
;training/Adam/gradients/loss/dense_13_loss/value_grad/Sum_1Sum9training/Adam/gradients/loss/dense_13_loss/value_grad/mulMtraining/Adam/gradients/loss/dense_13_loss/value_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
: 
ç
?training/Adam/gradients/loss/dense_13_loss/value_grad/Reshape_1Reshape;training/Adam/gradients/loss/dense_13_loss/value_grad/Sum_1=training/Adam/gradients/loss/dense_13_loss/value_grad/Shape_1*
T0*
_output_shapes
: 

Ctraining/Adam/gradients/loss/dense_13_loss/Sum_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 
í
=training/Adam/gradients/loss/dense_13_loss/Sum_1_grad/ReshapeReshape=training/Adam/gradients/loss/dense_13_loss/value_grad/ReshapeCtraining/Adam/gradients/loss/dense_13_loss/Sum_1_grad/Reshape/shape*
T0*
_output_shapes
: 
~
;training/Adam/gradients/loss/dense_13_loss/Sum_1_grad/ConstConst*
valueB *
dtype0*
_output_shapes
: 
ß
:training/Adam/gradients/loss/dense_13_loss/Sum_1_grad/TileTile=training/Adam/gradients/loss/dense_13_loss/Sum_1_grad/Reshape;training/Adam/gradients/loss/dense_13_loss/Sum_1_grad/Const*
T0*
_output_shapes
: 

Atraining/Adam/gradients/loss/dense_13_loss/Sum_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
ę
;training/Adam/gradients/loss/dense_13_loss/Sum_grad/ReshapeReshape:training/Adam/gradients/loss/dense_13_loss/Sum_1_grad/TileAtraining/Adam/gradients/loss/dense_13_loss/Sum_grad/Reshape/shape*
T0*
_output_shapes
:
­
9training/Adam/gradients/loss/dense_13_loss/Sum_grad/ShapeShapeDloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/Mul*
T0*
_output_shapes
:
ć
8training/Adam/gradients/loss/dense_13_loss/Sum_grad/TileTile;training/Adam/gradients/loss/dense_13_loss/Sum_grad/Reshape9training/Adam/gradients/loss/dense_13_loss/Sum_grad/Shape*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gtraining/Adam/gradients/loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/Mul_grad/ShapeShapezloss/dense_13_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*
_output_shapes
:
ë
itraining/Adam/gradients/loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/Mul_grad/Shape_1ShapeRloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights*
T0*
_output_shapes
:

wtraining/Adam/gradients/loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgsgtraining/Adam/gradients/loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/Mul_grad/Shapeitraining/Adam/gradients/loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/Mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
¨
etraining/Adam/gradients/loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/Mul_grad/MulMul8training/Adam/gradients/loss/dense_13_loss/Sum_grad/TileRloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ď
etraining/Adam/gradients/loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/Mul_grad/SumSumetraining/Adam/gradients/loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/Mul_grad/Mulwtraining/Adam/gradients/loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/Mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:
ň
itraining/Adam/gradients/loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/Mul_grad/ReshapeReshapeetraining/Adam/gradients/loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/Mul_grad/Sumgtraining/Adam/gradients/loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/Mul_grad/Shape*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ň
gtraining/Adam/gradients/loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/Mul_grad/Mul_1Mulzloss/dense_13_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits8training/Adam/gradients/loss/dense_13_loss/Sum_grad/Tile*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ő
gtraining/Adam/gradients/loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/Mul_grad/Sum_1Sumgtraining/Adam/gradients/loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/Mul_grad/Mul_1ytraining/Adam/gradients/loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/Mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
ř
ktraining/Adam/gradients/loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/Mul_grad/Reshape_1Reshapegtraining/Adam/gradients/loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/Mul_grad/Sum_1itraining/Adam/gradients/loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/Mul_grad/Shape_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ß
"training/Adam/gradients/zeros_like	ZerosLike|loss/dense_13_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

˘
§training/Adam/gradients/loss/dense_13_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradient|loss/dense_13_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*´
message¨ĽCurrently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

ň
Śtraining/Adam/gradients/loss/dense_13_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
÷
˘training/Adam/gradients/loss/dense_13_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDimsitraining/Adam/gradients/loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/Mul_grad/ReshapeŚtraining/Adam/gradients/loss/dense_13_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¤
training/Adam/gradients/loss/dense_13_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMul˘training/Adam/gradients/loss/dense_13_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims§training/Adam/gradients/loss/dense_13_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙


_training/Adam/gradients/loss/dense_13_loss/sparse_categorical_crossentropy/Reshape_1_grad/ShapeShapedense_13/BiasAdd*
T0*
_output_shapes
:

atraining/Adam/gradients/loss/dense_13_loss/sparse_categorical_crossentropy/Reshape_1_grad/ReshapeReshapetraining/Adam/gradients/loss/dense_13_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul_training/Adam/gradients/loss/dense_13_loss/sparse_categorical_crossentropy/Reshape_1_grad/Shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Đ
9training/Adam/gradients/dense_13/BiasAdd_grad/BiasAddGradBiasAddGradatraining/Adam/gradients/loss/dense_13_loss/sparse_categorical_crossentropy/Reshape_1_grad/Reshape*
T0*
_output_shapes
:


3training/Adam/gradients/dense_13/MatMul_grad/MatMulMatMulatraining/Adam/gradients/loss/dense_13_loss/sparse_categorical_crossentropy/Reshape_1_grad/Reshapedense_13/MatMul/ReadVariableOp*
transpose_b(*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ř
5training/Adam/gradients/dense_13/MatMul_grad/MatMul_1MatMuldropout_6/dropout/mul_1atraining/Adam/gradients/loss/dense_13_loss/sparse_categorical_crossentropy/Reshape_1_grad/Reshape*
T0*
_output_shapes
:	
*
transpose_a(

:training/Adam/gradients/dropout_6/dropout/mul_1_grad/ShapeShapedropout_6/dropout/mul*
T0*
_output_shapes
:

<training/Adam/gradients/dropout_6/dropout/mul_1_grad/Shape_1Shapedropout_6/dropout/Cast*
T0*
_output_shapes
:

Jtraining/Adam/gradients/dropout_6/dropout/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs:training/Adam/gradients/dropout_6/dropout/mul_1_grad/Shape<training/Adam/gradients/dropout_6/dropout/mul_1_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ż
8training/Adam/gradients/dropout_6/dropout/mul_1_grad/MulMul3training/Adam/gradients/dense_13/MatMul_grad/MatMuldropout_6/dropout/Cast*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
č
8training/Adam/gradients/dropout_6/dropout/mul_1_grad/SumSum8training/Adam/gradients/dropout_6/dropout/mul_1_grad/MulJtraining/Adam/gradients/dropout_6/dropout/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:
đ
<training/Adam/gradients/dropout_6/dropout/mul_1_grad/ReshapeReshape8training/Adam/gradients/dropout_6/dropout/mul_1_grad/Sum:training/Adam/gradients/dropout_6/dropout/mul_1_grad/Shape*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ŕ
:training/Adam/gradients/dropout_6/dropout/mul_1_grad/Mul_1Muldropout_6/dropout/mul3training/Adam/gradients/dense_13/MatMul_grad/MatMul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
î
:training/Adam/gradients/dropout_6/dropout/mul_1_grad/Sum_1Sum:training/Adam/gradients/dropout_6/dropout/mul_1_grad/Mul_1Ltraining/Adam/gradients/dropout_6/dropout/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
ö
>training/Adam/gradients/dropout_6/dropout/mul_1_grad/Reshape_1Reshape:training/Adam/gradients/dropout_6/dropout/mul_1_grad/Sum_1<training/Adam/gradients/dropout_6/dropout/mul_1_grad/Shape_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
u
8training/Adam/gradients/dropout_6/dropout/mul_grad/ShapeShapedense_12/Relu*
T0*
_output_shapes
:
}
:training/Adam/gradients/dropout_6/dropout/mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 

Htraining/Adam/gradients/dropout_6/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs8training/Adam/gradients/dropout_6/dropout/mul_grad/Shape:training/Adam/gradients/dropout_6/dropout/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
É
6training/Adam/gradients/dropout_6/dropout/mul_grad/MulMul<training/Adam/gradients/dropout_6/dropout/mul_1_grad/Reshapedropout_6/dropout/truediv*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
â
6training/Adam/gradients/dropout_6/dropout/mul_grad/SumSum6training/Adam/gradients/dropout_6/dropout/mul_grad/MulHtraining/Adam/gradients/dropout_6/dropout/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:
ę
:training/Adam/gradients/dropout_6/dropout/mul_grad/ReshapeReshape6training/Adam/gradients/dropout_6/dropout/mul_grad/Sum8training/Adam/gradients/dropout_6/dropout/mul_grad/Shape*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ż
8training/Adam/gradients/dropout_6/dropout/mul_grad/Mul_1Muldense_12/Relu<training/Adam/gradients/dropout_6/dropout/mul_1_grad/Reshape*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
č
8training/Adam/gradients/dropout_6/dropout/mul_grad/Sum_1Sum8training/Adam/gradients/dropout_6/dropout/mul_grad/Mul_1Jtraining/Adam/gradients/dropout_6/dropout/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
Ţ
<training/Adam/gradients/dropout_6/dropout/mul_grad/Reshape_1Reshape8training/Adam/gradients/dropout_6/dropout/mul_grad/Sum_1:training/Adam/gradients/dropout_6/dropout/mul_grad/Shape_1*
T0*
_output_shapes
: 
˝
3training/Adam/gradients/dense_12/Relu_grad/ReluGradReluGrad:training/Adam/gradients/dropout_6/dropout/mul_grad/Reshapedense_12/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
9training/Adam/gradients/dense_12/BiasAdd_grad/BiasAddGradBiasAddGrad3training/Adam/gradients/dense_12/Relu_grad/ReluGrad*
T0*
_output_shapes	
:
Ř
3training/Adam/gradients/dense_12/MatMul_grad/MatMulMatMul3training/Adam/gradients/dense_12/Relu_grad/ReluGraddense_12/MatMul/ReadVariableOp*
transpose_b(*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Â
5training/Adam/gradients/dense_12/MatMul_grad/MatMul_1MatMuldense_12_input3training/Adam/gradients/dense_12/Relu_grad/ReluGrad*
T0* 
_output_shapes
:
*
transpose_a(
Ć
Atraining/Adam/dense_12/kernel/m/Initializer/zeros/shape_as_tensorConst*2
_class(
&$loc:@training/Adam/dense_12/kernel/m*
valueB"     *
dtype0*
_output_shapes
:
°
7training/Adam/dense_12/kernel/m/Initializer/zeros/ConstConst*2
_class(
&$loc:@training/Adam/dense_12/kernel/m*
valueB
 *    *
dtype0*
_output_shapes
: 

1training/Adam/dense_12/kernel/m/Initializer/zerosFillAtraining/Adam/dense_12/kernel/m/Initializer/zeros/shape_as_tensor7training/Adam/dense_12/kernel/m/Initializer/zeros/Const*
T0*2
_class(
&$loc:@training/Adam/dense_12/kernel/m* 
_output_shapes
:

Đ
training/Adam/dense_12/kernel/mVarHandleOp*
shape:
*0
shared_name!training/Adam/dense_12/kernel/m*2
_class(
&$loc:@training/Adam/dense_12/kernel/m*
dtype0*
_output_shapes
: 

@training/Adam/dense_12/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_12/kernel/m*
_output_shapes
: 
Ď
&training/Adam/dense_12/kernel/m/AssignAssignVariableOptraining/Adam/dense_12/kernel/m1training/Adam/dense_12/kernel/m/Initializer/zeros*2
_class(
&$loc:@training/Adam/dense_12/kernel/m*
dtype0
É
3training/Adam/dense_12/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_12/kernel/m*2
_class(
&$loc:@training/Adam/dense_12/kernel/m*
dtype0* 
_output_shapes
:

°
/training/Adam/dense_12/bias/m/Initializer/zerosConst*0
_class&
$"loc:@training/Adam/dense_12/bias/m*
valueB*    *
dtype0*
_output_shapes	
:
Ĺ
training/Adam/dense_12/bias/mVarHandleOp*
shape:*.
shared_nametraining/Adam/dense_12/bias/m*0
_class&
$"loc:@training/Adam/dense_12/bias/m*
dtype0*
_output_shapes
: 

>training/Adam/dense_12/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_12/bias/m*
_output_shapes
: 
Ç
$training/Adam/dense_12/bias/m/AssignAssignVariableOptraining/Adam/dense_12/bias/m/training/Adam/dense_12/bias/m/Initializer/zeros*0
_class&
$"loc:@training/Adam/dense_12/bias/m*
dtype0
ž
1training/Adam/dense_12/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_12/bias/m*0
_class&
$"loc:@training/Adam/dense_12/bias/m*
dtype0*
_output_shapes	
:
Ć
Atraining/Adam/dense_13/kernel/m/Initializer/zeros/shape_as_tensorConst*2
_class(
&$loc:@training/Adam/dense_13/kernel/m*
valueB"   
   *
dtype0*
_output_shapes
:
°
7training/Adam/dense_13/kernel/m/Initializer/zeros/ConstConst*2
_class(
&$loc:@training/Adam/dense_13/kernel/m*
valueB
 *    *
dtype0*
_output_shapes
: 

1training/Adam/dense_13/kernel/m/Initializer/zerosFillAtraining/Adam/dense_13/kernel/m/Initializer/zeros/shape_as_tensor7training/Adam/dense_13/kernel/m/Initializer/zeros/Const*
T0*2
_class(
&$loc:@training/Adam/dense_13/kernel/m*
_output_shapes
:	

Ď
training/Adam/dense_13/kernel/mVarHandleOp*
shape:	
*0
shared_name!training/Adam/dense_13/kernel/m*2
_class(
&$loc:@training/Adam/dense_13/kernel/m*
dtype0*
_output_shapes
: 

@training/Adam/dense_13/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_13/kernel/m*
_output_shapes
: 
Ď
&training/Adam/dense_13/kernel/m/AssignAssignVariableOptraining/Adam/dense_13/kernel/m1training/Adam/dense_13/kernel/m/Initializer/zeros*2
_class(
&$loc:@training/Adam/dense_13/kernel/m*
dtype0
Č
3training/Adam/dense_13/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_13/kernel/m*2
_class(
&$loc:@training/Adam/dense_13/kernel/m*
dtype0*
_output_shapes
:	

Ž
/training/Adam/dense_13/bias/m/Initializer/zerosConst*0
_class&
$"loc:@training/Adam/dense_13/bias/m*
valueB
*    *
dtype0*
_output_shapes
:

Ä
training/Adam/dense_13/bias/mVarHandleOp*
shape:
*.
shared_nametraining/Adam/dense_13/bias/m*0
_class&
$"loc:@training/Adam/dense_13/bias/m*
dtype0*
_output_shapes
: 

>training/Adam/dense_13/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_13/bias/m*
_output_shapes
: 
Ç
$training/Adam/dense_13/bias/m/AssignAssignVariableOptraining/Adam/dense_13/bias/m/training/Adam/dense_13/bias/m/Initializer/zeros*0
_class&
$"loc:@training/Adam/dense_13/bias/m*
dtype0
˝
1training/Adam/dense_13/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_13/bias/m*0
_class&
$"loc:@training/Adam/dense_13/bias/m*
dtype0*
_output_shapes
:

Ć
Atraining/Adam/dense_12/kernel/v/Initializer/zeros/shape_as_tensorConst*2
_class(
&$loc:@training/Adam/dense_12/kernel/v*
valueB"     *
dtype0*
_output_shapes
:
°
7training/Adam/dense_12/kernel/v/Initializer/zeros/ConstConst*2
_class(
&$loc:@training/Adam/dense_12/kernel/v*
valueB
 *    *
dtype0*
_output_shapes
: 

1training/Adam/dense_12/kernel/v/Initializer/zerosFillAtraining/Adam/dense_12/kernel/v/Initializer/zeros/shape_as_tensor7training/Adam/dense_12/kernel/v/Initializer/zeros/Const*
T0*2
_class(
&$loc:@training/Adam/dense_12/kernel/v* 
_output_shapes
:

Đ
training/Adam/dense_12/kernel/vVarHandleOp*
shape:
*0
shared_name!training/Adam/dense_12/kernel/v*2
_class(
&$loc:@training/Adam/dense_12/kernel/v*
dtype0*
_output_shapes
: 

@training/Adam/dense_12/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_12/kernel/v*
_output_shapes
: 
Ď
&training/Adam/dense_12/kernel/v/AssignAssignVariableOptraining/Adam/dense_12/kernel/v1training/Adam/dense_12/kernel/v/Initializer/zeros*2
_class(
&$loc:@training/Adam/dense_12/kernel/v*
dtype0
É
3training/Adam/dense_12/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_12/kernel/v*2
_class(
&$loc:@training/Adam/dense_12/kernel/v*
dtype0* 
_output_shapes
:

°
/training/Adam/dense_12/bias/v/Initializer/zerosConst*0
_class&
$"loc:@training/Adam/dense_12/bias/v*
valueB*    *
dtype0*
_output_shapes	
:
Ĺ
training/Adam/dense_12/bias/vVarHandleOp*
shape:*.
shared_nametraining/Adam/dense_12/bias/v*0
_class&
$"loc:@training/Adam/dense_12/bias/v*
dtype0*
_output_shapes
: 

>training/Adam/dense_12/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_12/bias/v*
_output_shapes
: 
Ç
$training/Adam/dense_12/bias/v/AssignAssignVariableOptraining/Adam/dense_12/bias/v/training/Adam/dense_12/bias/v/Initializer/zeros*0
_class&
$"loc:@training/Adam/dense_12/bias/v*
dtype0
ž
1training/Adam/dense_12/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_12/bias/v*0
_class&
$"loc:@training/Adam/dense_12/bias/v*
dtype0*
_output_shapes	
:
Ć
Atraining/Adam/dense_13/kernel/v/Initializer/zeros/shape_as_tensorConst*2
_class(
&$loc:@training/Adam/dense_13/kernel/v*
valueB"   
   *
dtype0*
_output_shapes
:
°
7training/Adam/dense_13/kernel/v/Initializer/zeros/ConstConst*2
_class(
&$loc:@training/Adam/dense_13/kernel/v*
valueB
 *    *
dtype0*
_output_shapes
: 

1training/Adam/dense_13/kernel/v/Initializer/zerosFillAtraining/Adam/dense_13/kernel/v/Initializer/zeros/shape_as_tensor7training/Adam/dense_13/kernel/v/Initializer/zeros/Const*
T0*2
_class(
&$loc:@training/Adam/dense_13/kernel/v*
_output_shapes
:	

Ď
training/Adam/dense_13/kernel/vVarHandleOp*
shape:	
*0
shared_name!training/Adam/dense_13/kernel/v*2
_class(
&$loc:@training/Adam/dense_13/kernel/v*
dtype0*
_output_shapes
: 

@training/Adam/dense_13/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_13/kernel/v*
_output_shapes
: 
Ď
&training/Adam/dense_13/kernel/v/AssignAssignVariableOptraining/Adam/dense_13/kernel/v1training/Adam/dense_13/kernel/v/Initializer/zeros*2
_class(
&$loc:@training/Adam/dense_13/kernel/v*
dtype0
Č
3training/Adam/dense_13/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_13/kernel/v*2
_class(
&$loc:@training/Adam/dense_13/kernel/v*
dtype0*
_output_shapes
:	

Ž
/training/Adam/dense_13/bias/v/Initializer/zerosConst*0
_class&
$"loc:@training/Adam/dense_13/bias/v*
valueB
*    *
dtype0*
_output_shapes
:

Ä
training/Adam/dense_13/bias/vVarHandleOp*
shape:
*.
shared_nametraining/Adam/dense_13/bias/v*0
_class&
$"loc:@training/Adam/dense_13/bias/v*
dtype0*
_output_shapes
: 

>training/Adam/dense_13/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_13/bias/v*
_output_shapes
: 
Ç
$training/Adam/dense_13/bias/v/AssignAssignVariableOptraining/Adam/dense_13/bias/v/training/Adam/dense_13/bias/v/Initializer/zeros*0
_class&
$"loc:@training/Adam/dense_13/bias/v*
dtype0
˝
1training/Adam/dense_13/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_13/bias/v*0
_class&
$"loc:@training/Adam/dense_13/bias/v*
dtype0*
_output_shapes
:


8training/Adam/Adam/update_dense_12/kernel/ReadVariableOpReadVariableOpiter"/device:CPU:0*
dtype0	*
_output_shapes
: 

/training/Adam/Adam/update_dense_12/kernel/add/yConst*"
_class
loc:@dense_12/kernel*
value	B	 R*
dtype0	*
_output_shapes
: 
ä
-training/Adam/Adam/update_dense_12/kernel/addAdd8training/Adam/Adam/update_dense_12/kernel/ReadVariableOp/training/Adam/Adam/update_dense_12/kernel/add/y*
T0	*"
_class
loc:@dense_12/kernel*
_output_shapes
: 
š
.training/Adam/Adam/update_dense_12/kernel/CastCast-training/Adam/Adam/update_dense_12/kernel/add*

SrcT0	*"
_class
loc:@dense_12/kernel*
_output_shapes
: *

DstT0
{
<training/Adam/Adam/update_dense_12/kernel/Pow/ReadVariableOpReadVariableOpbeta_1*
dtype0*
_output_shapes
: 
ç
-training/Adam/Adam/update_dense_12/kernel/PowPow<training/Adam/Adam/update_dense_12/kernel/Pow/ReadVariableOp.training/Adam/Adam/update_dense_12/kernel/Cast*
T0*"
_class
loc:@dense_12/kernel*
_output_shapes
: 
}
>training/Adam/Adam/update_dense_12/kernel/Pow_1/ReadVariableOpReadVariableOpbeta_2*
dtype0*
_output_shapes
: 
ë
/training/Adam/Adam/update_dense_12/kernel/Pow_1Pow>training/Adam/Adam/update_dense_12/kernel/Pow_1/ReadVariableOp.training/Adam/Adam/update_dense_12/kernel/Cast*
T0*"
_class
loc:@dense_12/kernel*
_output_shapes
: 

Jtraining/Adam/Adam/update_dense_12/kernel/ResourceApplyAdam/ReadVariableOpReadVariableOplearning_rate*
dtype0*
_output_shapes
: 

Ltraining/Adam/Adam/update_dense_12/kernel/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta_1*
dtype0*
_output_shapes
: 

Ltraining/Adam/Adam/update_dense_12/kernel/ResourceApplyAdam/ReadVariableOp_2ReadVariableOpbeta_2*
dtype0*
_output_shapes
: 

Ltraining/Adam/Adam/update_dense_12/kernel/ResourceApplyAdam/ReadVariableOp_3ReadVariableOpepsilon*
dtype0*
_output_shapes
: 
°
;training/Adam/Adam/update_dense_12/kernel/ResourceApplyAdamResourceApplyAdamdense_12/kerneltraining/Adam/dense_12/kernel/mtraining/Adam/dense_12/kernel/v-training/Adam/Adam/update_dense_12/kernel/Pow/training/Adam/Adam/update_dense_12/kernel/Pow_1Jtraining/Adam/Adam/update_dense_12/kernel/ResourceApplyAdam/ReadVariableOpLtraining/Adam/Adam/update_dense_12/kernel/ResourceApplyAdam/ReadVariableOp_1Ltraining/Adam/Adam/update_dense_12/kernel/ResourceApplyAdam/ReadVariableOp_2Ltraining/Adam/Adam/update_dense_12/kernel/ResourceApplyAdam/ReadVariableOp_35training/Adam/gradients/dense_12/MatMul_grad/MatMul_1*
use_locking(*
T0*"
_class
loc:@dense_12/kernel

6training/Adam/Adam/update_dense_12/bias/ReadVariableOpReadVariableOpiter"/device:CPU:0*
dtype0	*
_output_shapes
: 

-training/Adam/Adam/update_dense_12/bias/add/yConst* 
_class
loc:@dense_12/bias*
value	B	 R*
dtype0	*
_output_shapes
: 
Ü
+training/Adam/Adam/update_dense_12/bias/addAdd6training/Adam/Adam/update_dense_12/bias/ReadVariableOp-training/Adam/Adam/update_dense_12/bias/add/y*
T0	* 
_class
loc:@dense_12/bias*
_output_shapes
: 
ł
,training/Adam/Adam/update_dense_12/bias/CastCast+training/Adam/Adam/update_dense_12/bias/add*

SrcT0	* 
_class
loc:@dense_12/bias*
_output_shapes
: *

DstT0
y
:training/Adam/Adam/update_dense_12/bias/Pow/ReadVariableOpReadVariableOpbeta_1*
dtype0*
_output_shapes
: 
ß
+training/Adam/Adam/update_dense_12/bias/PowPow:training/Adam/Adam/update_dense_12/bias/Pow/ReadVariableOp,training/Adam/Adam/update_dense_12/bias/Cast*
T0* 
_class
loc:@dense_12/bias*
_output_shapes
: 
{
<training/Adam/Adam/update_dense_12/bias/Pow_1/ReadVariableOpReadVariableOpbeta_2*
dtype0*
_output_shapes
: 
ă
-training/Adam/Adam/update_dense_12/bias/Pow_1Pow<training/Adam/Adam/update_dense_12/bias/Pow_1/ReadVariableOp,training/Adam/Adam/update_dense_12/bias/Cast*
T0* 
_class
loc:@dense_12/bias*
_output_shapes
: 

Htraining/Adam/Adam/update_dense_12/bias/ResourceApplyAdam/ReadVariableOpReadVariableOplearning_rate*
dtype0*
_output_shapes
: 

Jtraining/Adam/Adam/update_dense_12/bias/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta_1*
dtype0*
_output_shapes
: 

Jtraining/Adam/Adam/update_dense_12/bias/ResourceApplyAdam/ReadVariableOp_2ReadVariableOpbeta_2*
dtype0*
_output_shapes
: 

Jtraining/Adam/Adam/update_dense_12/bias/ResourceApplyAdam/ReadVariableOp_3ReadVariableOpepsilon*
dtype0*
_output_shapes
: 

9training/Adam/Adam/update_dense_12/bias/ResourceApplyAdamResourceApplyAdamdense_12/biastraining/Adam/dense_12/bias/mtraining/Adam/dense_12/bias/v+training/Adam/Adam/update_dense_12/bias/Pow-training/Adam/Adam/update_dense_12/bias/Pow_1Htraining/Adam/Adam/update_dense_12/bias/ResourceApplyAdam/ReadVariableOpJtraining/Adam/Adam/update_dense_12/bias/ResourceApplyAdam/ReadVariableOp_1Jtraining/Adam/Adam/update_dense_12/bias/ResourceApplyAdam/ReadVariableOp_2Jtraining/Adam/Adam/update_dense_12/bias/ResourceApplyAdam/ReadVariableOp_39training/Adam/gradients/dense_12/BiasAdd_grad/BiasAddGrad*
use_locking(*
T0* 
_class
loc:@dense_12/bias

8training/Adam/Adam/update_dense_13/kernel/ReadVariableOpReadVariableOpiter"/device:CPU:0*
dtype0	*
_output_shapes
: 

/training/Adam/Adam/update_dense_13/kernel/add/yConst*"
_class
loc:@dense_13/kernel*
value	B	 R*
dtype0	*
_output_shapes
: 
ä
-training/Adam/Adam/update_dense_13/kernel/addAdd8training/Adam/Adam/update_dense_13/kernel/ReadVariableOp/training/Adam/Adam/update_dense_13/kernel/add/y*
T0	*"
_class
loc:@dense_13/kernel*
_output_shapes
: 
š
.training/Adam/Adam/update_dense_13/kernel/CastCast-training/Adam/Adam/update_dense_13/kernel/add*

SrcT0	*"
_class
loc:@dense_13/kernel*
_output_shapes
: *

DstT0
{
<training/Adam/Adam/update_dense_13/kernel/Pow/ReadVariableOpReadVariableOpbeta_1*
dtype0*
_output_shapes
: 
ç
-training/Adam/Adam/update_dense_13/kernel/PowPow<training/Adam/Adam/update_dense_13/kernel/Pow/ReadVariableOp.training/Adam/Adam/update_dense_13/kernel/Cast*
T0*"
_class
loc:@dense_13/kernel*
_output_shapes
: 
}
>training/Adam/Adam/update_dense_13/kernel/Pow_1/ReadVariableOpReadVariableOpbeta_2*
dtype0*
_output_shapes
: 
ë
/training/Adam/Adam/update_dense_13/kernel/Pow_1Pow>training/Adam/Adam/update_dense_13/kernel/Pow_1/ReadVariableOp.training/Adam/Adam/update_dense_13/kernel/Cast*
T0*"
_class
loc:@dense_13/kernel*
_output_shapes
: 

Jtraining/Adam/Adam/update_dense_13/kernel/ResourceApplyAdam/ReadVariableOpReadVariableOplearning_rate*
dtype0*
_output_shapes
: 

Ltraining/Adam/Adam/update_dense_13/kernel/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta_1*
dtype0*
_output_shapes
: 

Ltraining/Adam/Adam/update_dense_13/kernel/ResourceApplyAdam/ReadVariableOp_2ReadVariableOpbeta_2*
dtype0*
_output_shapes
: 

Ltraining/Adam/Adam/update_dense_13/kernel/ResourceApplyAdam/ReadVariableOp_3ReadVariableOpepsilon*
dtype0*
_output_shapes
: 
°
;training/Adam/Adam/update_dense_13/kernel/ResourceApplyAdamResourceApplyAdamdense_13/kerneltraining/Adam/dense_13/kernel/mtraining/Adam/dense_13/kernel/v-training/Adam/Adam/update_dense_13/kernel/Pow/training/Adam/Adam/update_dense_13/kernel/Pow_1Jtraining/Adam/Adam/update_dense_13/kernel/ResourceApplyAdam/ReadVariableOpLtraining/Adam/Adam/update_dense_13/kernel/ResourceApplyAdam/ReadVariableOp_1Ltraining/Adam/Adam/update_dense_13/kernel/ResourceApplyAdam/ReadVariableOp_2Ltraining/Adam/Adam/update_dense_13/kernel/ResourceApplyAdam/ReadVariableOp_35training/Adam/gradients/dense_13/MatMul_grad/MatMul_1*
use_locking(*
T0*"
_class
loc:@dense_13/kernel

6training/Adam/Adam/update_dense_13/bias/ReadVariableOpReadVariableOpiter"/device:CPU:0*
dtype0	*
_output_shapes
: 

-training/Adam/Adam/update_dense_13/bias/add/yConst* 
_class
loc:@dense_13/bias*
value	B	 R*
dtype0	*
_output_shapes
: 
Ü
+training/Adam/Adam/update_dense_13/bias/addAdd6training/Adam/Adam/update_dense_13/bias/ReadVariableOp-training/Adam/Adam/update_dense_13/bias/add/y*
T0	* 
_class
loc:@dense_13/bias*
_output_shapes
: 
ł
,training/Adam/Adam/update_dense_13/bias/CastCast+training/Adam/Adam/update_dense_13/bias/add*

SrcT0	* 
_class
loc:@dense_13/bias*
_output_shapes
: *

DstT0
y
:training/Adam/Adam/update_dense_13/bias/Pow/ReadVariableOpReadVariableOpbeta_1*
dtype0*
_output_shapes
: 
ß
+training/Adam/Adam/update_dense_13/bias/PowPow:training/Adam/Adam/update_dense_13/bias/Pow/ReadVariableOp,training/Adam/Adam/update_dense_13/bias/Cast*
T0* 
_class
loc:@dense_13/bias*
_output_shapes
: 
{
<training/Adam/Adam/update_dense_13/bias/Pow_1/ReadVariableOpReadVariableOpbeta_2*
dtype0*
_output_shapes
: 
ă
-training/Adam/Adam/update_dense_13/bias/Pow_1Pow<training/Adam/Adam/update_dense_13/bias/Pow_1/ReadVariableOp,training/Adam/Adam/update_dense_13/bias/Cast*
T0* 
_class
loc:@dense_13/bias*
_output_shapes
: 

Htraining/Adam/Adam/update_dense_13/bias/ResourceApplyAdam/ReadVariableOpReadVariableOplearning_rate*
dtype0*
_output_shapes
: 

Jtraining/Adam/Adam/update_dense_13/bias/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta_1*
dtype0*
_output_shapes
: 

Jtraining/Adam/Adam/update_dense_13/bias/ResourceApplyAdam/ReadVariableOp_2ReadVariableOpbeta_2*
dtype0*
_output_shapes
: 

Jtraining/Adam/Adam/update_dense_13/bias/ResourceApplyAdam/ReadVariableOp_3ReadVariableOpepsilon*
dtype0*
_output_shapes
: 

9training/Adam/Adam/update_dense_13/bias/ResourceApplyAdamResourceApplyAdamdense_13/biastraining/Adam/dense_13/bias/mtraining/Adam/dense_13/bias/v+training/Adam/Adam/update_dense_13/bias/Pow-training/Adam/Adam/update_dense_13/bias/Pow_1Htraining/Adam/Adam/update_dense_13/bias/ResourceApplyAdam/ReadVariableOpJtraining/Adam/Adam/update_dense_13/bias/ResourceApplyAdam/ReadVariableOp_1Jtraining/Adam/Adam/update_dense_13/bias/ResourceApplyAdam/ReadVariableOp_2Jtraining/Adam/Adam/update_dense_13/bias/ResourceApplyAdam/ReadVariableOp_39training/Adam/gradients/dense_13/BiasAdd_grad/BiasAddGrad*
use_locking(*
T0* 
_class
loc:@dense_13/bias
Î
training/Adam/Adam/ConstConst:^training/Adam/Adam/update_dense_12/bias/ResourceApplyAdam<^training/Adam/Adam/update_dense_12/kernel/ResourceApplyAdam:^training/Adam/Adam/update_dense_13/bias/ResourceApplyAdam<^training/Adam/Adam/update_dense_13/kernel/ResourceApplyAdam*
value	B	 R*
dtype0	*
_output_shapes
: 
j
&training/Adam/Adam/AssignAddVariableOpAssignAddVariableOpitertraining/Adam/Adam/Const*
dtype0	
ű
!training/Adam/Adam/ReadVariableOpReadVariableOpiter'^training/Adam/Adam/AssignAddVariableOp:^training/Adam/Adam/update_dense_12/bias/ResourceApplyAdam<^training/Adam/Adam/update_dense_12/kernel/ResourceApplyAdam:^training/Adam/Adam/update_dense_13/bias/ResourceApplyAdam<^training/Adam/Adam/update_dense_13/kernel/ResourceApplyAdam*
dtype0	*
_output_shapes
: 
i
training_1/group_depsNoOp	^loss/mul^metrics/accuracy/Mean'^training/Adam/Adam/AssignAddVariableOp
W
Const_1Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_2Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_3Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_4Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_5Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
G
VarIsInitializedOpVarIsInitializedOpdecay*
_output_shapes
: 
Q
VarIsInitializedOp_1VarIsInitializedOpdense_12/bias*
_output_shapes
: 
c
VarIsInitializedOp_2VarIsInitializedOptraining/Adam/dense_12/kernel/m*
_output_shapes
: 
S
VarIsInitializedOp_3VarIsInitializedOpdense_13/kernel*
_output_shapes
: 
a
VarIsInitializedOp_4VarIsInitializedOptraining/Adam/dense_12/bias/m*
_output_shapes
: 
c
VarIsInitializedOp_5VarIsInitializedOptraining/Adam/dense_13/kernel/m*
_output_shapes
: 
I
VarIsInitializedOp_6VarIsInitializedOptotal*
_output_shapes
: 
a
VarIsInitializedOp_7VarIsInitializedOptraining/Adam/dense_13/bias/m*
_output_shapes
: 
Q
VarIsInitializedOp_8VarIsInitializedOplearning_rate*
_output_shapes
: 
c
VarIsInitializedOp_9VarIsInitializedOptraining/Adam/dense_12/kernel/v*
_output_shapes
: 
R
VarIsInitializedOp_10VarIsInitializedOpdense_13/bias*
_output_shapes
: 
b
VarIsInitializedOp_11VarIsInitializedOptraining/Adam/dense_12/bias/v*
_output_shapes
: 
L
VarIsInitializedOp_12VarIsInitializedOpepsilon*
_output_shapes
: 
J
VarIsInitializedOp_13VarIsInitializedOpcount*
_output_shapes
: 
d
VarIsInitializedOp_14VarIsInitializedOptraining/Adam/dense_13/kernel/v*
_output_shapes
: 
T
VarIsInitializedOp_15VarIsInitializedOpdense_12/kernel*
_output_shapes
: 
K
VarIsInitializedOp_16VarIsInitializedOpbeta_2*
_output_shapes
: 
b
VarIsInitializedOp_17VarIsInitializedOptraining/Adam/dense_13/bias/v*
_output_shapes
: 
I
VarIsInitializedOp_18VarIsInitializedOpiter*
_output_shapes
: 
K
VarIsInitializedOp_19VarIsInitializedOpbeta_1*
_output_shapes
: 
Ś
	init/NoOpNoOp^beta_1/Assign^beta_2/Assign^count/Assign^decay/Assign^dense_12/bias/Assign^dense_12/kernel/Assign^dense_13/bias/Assign^dense_13/kernel/Assign^epsilon/Assign^learning_rate/Assign^total/Assign%^training/Adam/dense_12/bias/m/Assign%^training/Adam/dense_12/bias/v/Assign'^training/Adam/dense_12/kernel/m/Assign'^training/Adam/dense_12/kernel/v/Assign%^training/Adam/dense_13/bias/m/Assign%^training/Adam/dense_13/bias/v/Assign'^training/Adam/dense_13/kernel/m/Assign'^training/Adam/dense_13/kernel/v/Assign
0
init/NoOp_1NoOp^iter/Assign"/device:CPU:0
&
initNoOp
^init/NoOp^init/NoOp_1
W
Const_6Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
\
Const_7Const"/device:CPU:0*
valueB Bmodel*
dtype0*
_output_shapes
: 
W
Const_8Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_9Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_10Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
Ŕ
RestoreV2/tensor_namesConst"/device:CPU:0*g
value^B\BRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
r
RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

	RestoreV2	RestoreV2Const_7RestoreV2/tensor_namesRestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
B
IdentityIdentity	RestoreV2*
T0*
_output_shapes
:
\
AssignVariableOpAssignVariableOptraining/Adam/dense_12/kernel/mIdentity*
dtype0
Â
RestoreV2_1/tensor_namesConst"/device:CPU:0*g
value^B\BRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_1	RestoreV2Const_7RestoreV2_1/tensor_namesRestoreV2_1/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
F

Identity_1IdentityRestoreV2_1*
T0*
_output_shapes
:
`
AssignVariableOp_1AssignVariableOptraining/Adam/dense_12/kernel/v
Identity_1*
dtype0
Ŕ
RestoreV2_2/tensor_namesConst"/device:CPU:0*e
value\BZBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_2/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_2	RestoreV2Const_7RestoreV2_2/tensor_namesRestoreV2_2/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
F

Identity_2IdentityRestoreV2_2*
T0*
_output_shapes
:
^
AssignVariableOp_2AssignVariableOptraining/Adam/dense_12/bias/m
Identity_2*
dtype0
Ŕ
RestoreV2_3/tensor_namesConst"/device:CPU:0*e
value\BZBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_3/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_3	RestoreV2Const_7RestoreV2_3/tensor_namesRestoreV2_3/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
F

Identity_3IdentityRestoreV2_3*
T0*
_output_shapes
:
^
AssignVariableOp_3AssignVariableOptraining/Adam/dense_12/bias/v
Identity_3*
dtype0
X
Const_11Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_12Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
Â
RestoreV2_4/tensor_namesConst"/device:CPU:0*g
value^B\BRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_4/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_4	RestoreV2Const_7RestoreV2_4/tensor_namesRestoreV2_4/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
F

Identity_4IdentityRestoreV2_4*
T0*
_output_shapes
:
`
AssignVariableOp_4AssignVariableOptraining/Adam/dense_13/kernel/m
Identity_4*
dtype0
Â
RestoreV2_5/tensor_namesConst"/device:CPU:0*g
value^B\BRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_5/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_5	RestoreV2Const_7RestoreV2_5/tensor_namesRestoreV2_5/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
F

Identity_5IdentityRestoreV2_5*
T0*
_output_shapes
:
`
AssignVariableOp_5AssignVariableOptraining/Adam/dense_13/kernel/v
Identity_5*
dtype0
Ŕ
RestoreV2_6/tensor_namesConst"/device:CPU:0*e
value\BZBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_6/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_6	RestoreV2Const_7RestoreV2_6/tensor_namesRestoreV2_6/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
F

Identity_6IdentityRestoreV2_6*
T0*
_output_shapes
:
^
AssignVariableOp_6AssignVariableOptraining/Adam/dense_13/bias/m
Identity_6*
dtype0
Ŕ
RestoreV2_7/tensor_namesConst"/device:CPU:0*e
value\BZBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_7/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_7	RestoreV2Const_7RestoreV2_7/tensor_namesRestoreV2_7/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
F

Identity_7IdentityRestoreV2_7*
T0*
_output_shapes
:
^
AssignVariableOp_7AssignVariableOptraining/Adam/dense_13/bias/v
Identity_7*
dtype0
X
Const_13Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
Ś
RestoreV2_8/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_8/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_8	RestoreV2Const_7RestoreV2_8/tensor_namesRestoreV2_8/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
F

Identity_8IdentityRestoreV2_8*
T0*
_output_shapes
:
P
AssignVariableOp_8AssignVariableOpdense_12/kernel
Identity_8*
dtype0
¤
RestoreV2_9/tensor_namesConst"/device:CPU:0*I
value@B>B4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_9/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_9	RestoreV2Const_7RestoreV2_9/tensor_namesRestoreV2_9/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
F

Identity_9IdentityRestoreV2_9*
T0*
_output_shapes
:
N
AssignVariableOp_9AssignVariableOpdense_12/bias
Identity_9*
dtype0
§
RestoreV2_10/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_10/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_10	RestoreV2Const_7RestoreV2_10/tensor_namesRestoreV2_10/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_10IdentityRestoreV2_10*
T0*
_output_shapes
:
R
AssignVariableOp_10AssignVariableOpdense_13/kernelIdentity_10*
dtype0
Ľ
RestoreV2_11/tensor_namesConst"/device:CPU:0*I
value@B>B4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_11/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_11	RestoreV2Const_7RestoreV2_11/tensor_namesRestoreV2_11/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_11IdentityRestoreV2_11*
T0*
_output_shapes
:
P
AssignVariableOp_11AssignVariableOpdense_13/biasIdentity_11*
dtype0

RestoreV2_12/tensor_namesConst"/device:CPU:0*>
value5B3B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_12/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_12	RestoreV2Const_7RestoreV2_12/tensor_namesRestoreV2_12/shape_and_slices"/device:CPU:0*
dtypes
2	*
_output_shapes
:
W
Identity_12IdentityRestoreV2_12"/device:CPU:0*
T0	*
_output_shapes
:
V
AssignVariableOp_12AssignVariableOpiterIdentity_12"/device:CPU:0*
dtype0	
Ł
RestoreV2_13/tensor_namesConst"/device:CPU:0*G
value>B<B2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_13/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_13	RestoreV2Const_7RestoreV2_13/tensor_namesRestoreV2_13/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_13IdentityRestoreV2_13*
T0*
_output_shapes
:
P
AssignVariableOp_13AssignVariableOplearning_rateIdentity_13*
dtype0

RestoreV2_14/tensor_namesConst"/device:CPU:0*?
value6B4B*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_14/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_14	RestoreV2Const_7RestoreV2_14/tensor_namesRestoreV2_14/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_14IdentityRestoreV2_14*
T0*
_output_shapes
:
H
AssignVariableOp_14AssignVariableOpdecayIdentity_14*
dtype0

RestoreV2_15/tensor_namesConst"/device:CPU:0*@
value7B5B+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_15/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_15	RestoreV2Const_7RestoreV2_15/tensor_namesRestoreV2_15/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_15IdentityRestoreV2_15*
T0*
_output_shapes
:
I
AssignVariableOp_15AssignVariableOpbeta_1Identity_15*
dtype0

RestoreV2_16/tensor_namesConst"/device:CPU:0*@
value7B5B+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_16/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_16	RestoreV2Const_7RestoreV2_16/tensor_namesRestoreV2_16/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_16IdentityRestoreV2_16*
T0*
_output_shapes
:
I
AssignVariableOp_16AssignVariableOpbeta_2Identity_16*
dtype0

RestoreV2_17/tensor_namesConst"/device:CPU:0*A
value8B6B,optimizer/epsilon/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_17/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_17	RestoreV2Const_7RestoreV2_17/tensor_namesRestoreV2_17/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_17IdentityRestoreV2_17*
T0*
_output_shapes
:
J
AssignVariableOp_17AssignVariableOpepsilonIdentity_17*
dtype0
X
Const_14Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_15Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 

SaveV2/tensor_namesConst"/device:CPU:0*ż
valueľB˛B/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB3layer_with_weights-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-2/.ATTRIBUTES/OBJECT_CONFIG_JSONB3layer_with_weights-1/.ATTRIBUTES/OBJECT_CONFIG_JSONB(optimizer/.ATTRIBUTES/OBJECT_CONFIG_JSONB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB,optimizer/epsilon/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:

SaveV2/shape_and_slicesConst"/device:CPU:0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
Ą
SaveV2SaveV2Const_15SaveV2/tensor_namesSaveV2/shape_and_slicesConst_8Const_9Const_10Const_11Const_12Const_13#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOpiter/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpdecay/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpepsilon/Read/ReadVariableOp3training/Adam/dense_12/kernel/m/Read/ReadVariableOp1training/Adam/dense_12/bias/m/Read/ReadVariableOp3training/Adam/dense_13/kernel/m/Read/ReadVariableOp1training/Adam/dense_13/bias/m/Read/ReadVariableOp3training/Adam/dense_12/kernel/v/Read/ReadVariableOp1training/Adam/dense_12/bias/v/Read/ReadVariableOp3training/Adam/dense_13/kernel/v/Read/ReadVariableOp1training/Adam/dense_13/bias/v/Read/ReadVariableOpConst_14"/device:CPU:0*'
dtypes
2	
Z
Identity_18IdentityConst_15^SaveV2"/device:CPU:0*
T0*
_output_shapes
: 
z
total_1/Initializer/zerosConst*
_class
loc:@total_1*
valueB
 *    *
dtype0*
_output_shapes
: 
~
total_1VarHandleOp*
shape: *
shared_name	total_1*
_class
loc:@total_1*
dtype0*
_output_shapes
: 
_
(total_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptotal_1*
_output_shapes
: 
o
total_1/AssignAssignVariableOptotal_1total_1/Initializer/zeros*
_class
loc:@total_1*
dtype0
w
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_class
loc:@total_1*
dtype0*
_output_shapes
: 
z
count_1/Initializer/zerosConst*
_class
loc:@count_1*
valueB
 *    *
dtype0*
_output_shapes
: 
~
count_1VarHandleOp*
shape: *
shared_name	count_1*
_class
loc:@count_1*
dtype0*
_output_shapes
: 
_
(count_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpcount_1*
_output_shapes
: 
o
count_1/AssignAssignVariableOpcount_1count_1/Initializer/zeros*
_class
loc:@count_1*
dtype0
w
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_class
loc:@count_1*
dtype0*
_output_shapes
: 
K
Const_16Const*
valueB *
dtype0*
_output_shapes
: 
L
SumSummetrics/accuracy/MeanConst_16*
T0*
_output_shapes
: 
E
AssignAddVariableOpAssignAddVariableOptotal_1Sum*
dtype0
j
ReadVariableOpReadVariableOptotal_1^AssignAddVariableOp^Sum*
dtype0*
_output_shapes
: 
F
SizeConst*
value	B :*
dtype0*
_output_shapes
: 
B
CastCastSize*

SrcT0*
_output_shapes
: *

DstT0
^
AssignAddVariableOp_1AssignAddVariableOpcount_1Cast^AssignAddVariableOp*
dtype0
~
ReadVariableOp_1ReadVariableOpcount_1^AssignAddVariableOp^AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
q
div_no_nan/ReadVariableOpReadVariableOptotal_1^AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
s
div_no_nan/ReadVariableOp_1ReadVariableOpcount_1^AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
o

div_no_nanDivNoNandiv_no_nan/ReadVariableOpdiv_no_nan/ReadVariableOp_1*
T0*
_output_shapes
: 
D
Identity_19Identity
div_no_nan*
T0*
_output_shapes
: 
[
div_no_nan_1/ReadVariableOpReadVariableOptotal_1*
dtype0*
_output_shapes
: 
]
div_no_nan_1/ReadVariableOp_1ReadVariableOpcount_1*
dtype0*
_output_shapes
: 
u
div_no_nan_1DivNoNandiv_no_nan_1/ReadVariableOpdiv_no_nan_1/ReadVariableOp_1*
T0*
_output_shapes
: 
F
Identity_20Identitydiv_no_nan_1*
T0*
_output_shapes
: 
l
metric_op_wrapperConst^AssignAddVariableOp_1*
valueB *
dtype0*
_output_shapes
: 
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
dtype0*
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
dtype0*
_output_shapes
: 
¨	
save/Const_1Const*ë
valueáBŢ B×{"class_name": "Sequential", "config": {"layers": [{"class_name": "Dense", "config": {"activation": "relu", "activity_regularizer": null, "batch_input_shape": [null, 784], "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "dtype": "float32", "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "kernel_regularizer": null, "name": "dense_12", "trainable": true, "units": 512, "use_bias": true}}, {"class_name": "Dropout", "config": {"dtype": "float32", "name": "dropout_6", "noise_shape": null, "rate": 0.2, "seed": null, "trainable": true}}, {"class_name": "Dense", "config": {"activation": "softmax", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "dtype": "float32", "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "kernel_regularizer": null, "name": "dense_13", "trainable": true, "units": 10, "use_bias": true}}], "name": "sequential_6"}}*
dtype0*
_output_shapes
: 
Ú
save/Const_2Const*
valueB B{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 784], "dtype": "float32", "name": "dense_12_input", "sparse": false}}*
dtype0*
_output_shapes
: 
â
save/Const_3Const*Ľ
valueB B{"class_name": "Dropout", "config": {"dtype": "float32", "name": "dropout_6", "noise_shape": null, "rate": 0.2, "seed": null, "trainable": true}}*
dtype0*
_output_shapes
: 

save/Const_4Const*Ţ
valueÔBŃ BĘ{"class_name": "Dense", "config": {"activation": "relu", "activity_regularizer": null, "batch_input_shape": [null, 784], "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "dtype": "float32", "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "kernel_regularizer": null, "name": "dense_12", "trainable": true, "units": 512, "use_bias": true}}*
dtype0*
_output_shapes
: 
ű
save/Const_5Const*ž
value´Bą BŞ{"class_name": "Dense", "config": {"activation": "softmax", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "dtype": "float32", "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "kernel_regularizer": null, "name": "dense_13", "trainable": true, "units": 10, "use_bias": true}}*
dtype0*
_output_shapes
: 
N
save/VarIsInitializedOpVarIsInitializedOptotal_1*
_output_shapes
: 
P
save/VarIsInitializedOp_1VarIsInitializedOpcount_1*
_output_shapes
: 
3
	save/initNoOp^count_1/Assign^total_1/Assign
Ş
save/Const_6Const*í
valueăBŕ BŮ{"class_name": "Adam", "config": {"amsgrad": false, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "decay": 0.0, "epsilon": 1.0000000116860974e-07, "learning_rate": 0.0010000000474974513, "name": "Adam"}}*
dtype0*
_output_shapes
: 
î
save/SaveV2/tensor_namesConst*Ą
valueBB/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-2/.ATTRIBUTES/OBJECT_CONFIG_JSONB3layer_with_weights-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-1/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/.ATTRIBUTES/OBJECT_CONFIG_JSONB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB,optimizer/epsilon/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:

save/SaveV2/shape_and_slicesConst*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
˛
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicessave/Const_1save/Const_2save/Const_3save/Const_4!dense_12/bias/Read/ReadVariableOp1training/Adam/dense_12/bias/m/Read/ReadVariableOp1training/Adam/dense_12/bias/v/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp3training/Adam/dense_12/kernel/m/Read/ReadVariableOp3training/Adam/dense_12/kernel/v/Read/ReadVariableOpsave/Const_5!dense_13/bias/Read/ReadVariableOp1training/Adam/dense_13/bias/m/Read/ReadVariableOp1training/Adam/dense_13/bias/v/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp3training/Adam/dense_13/kernel/m/Read/ReadVariableOp3training/Adam/dense_13/kernel/v/Read/ReadVariableOpsave/Const_6beta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOpepsilon/Read/ReadVariableOpiter/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp*&
dtypes
2	
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 

save/RestoreV2/tensor_namesConst"/device:CPU:0*Ą
valueBB/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-2/.ATTRIBUTES/OBJECT_CONFIG_JSONB3layer_with_weights-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-1/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/.ATTRIBUTES/OBJECT_CONFIG_JSONB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB,optimizer/epsilon/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
Ľ
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*&
dtypes
2	*t
_output_shapesb
`::::::::::::::::::::::::

	save/NoOpNoOp

save/NoOp_1NoOp

save/NoOp_2NoOp

save/NoOp_3NoOp
N
save/IdentityIdentitysave/RestoreV2:4*
T0*
_output_shapes
:
T
save/AssignVariableOpAssignVariableOpdense_12/biassave/Identity*
dtype0
P
save/Identity_1Identitysave/RestoreV2:5*
T0*
_output_shapes
:
h
save/AssignVariableOp_1AssignVariableOptraining/Adam/dense_12/bias/msave/Identity_1*
dtype0
P
save/Identity_2Identitysave/RestoreV2:6*
T0*
_output_shapes
:
h
save/AssignVariableOp_2AssignVariableOptraining/Adam/dense_12/bias/vsave/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:7*
T0*
_output_shapes
:
Z
save/AssignVariableOp_3AssignVariableOpdense_12/kernelsave/Identity_3*
dtype0
P
save/Identity_4Identitysave/RestoreV2:8*
T0*
_output_shapes
:
j
save/AssignVariableOp_4AssignVariableOptraining/Adam/dense_12/kernel/msave/Identity_4*
dtype0
P
save/Identity_5Identitysave/RestoreV2:9*
T0*
_output_shapes
:
j
save/AssignVariableOp_5AssignVariableOptraining/Adam/dense_12/kernel/vsave/Identity_5*
dtype0

save/NoOp_4NoOp
Q
save/Identity_6Identitysave/RestoreV2:11*
T0*
_output_shapes
:
X
save/AssignVariableOp_6AssignVariableOpdense_13/biassave/Identity_6*
dtype0
Q
save/Identity_7Identitysave/RestoreV2:12*
T0*
_output_shapes
:
h
save/AssignVariableOp_7AssignVariableOptraining/Adam/dense_13/bias/msave/Identity_7*
dtype0
Q
save/Identity_8Identitysave/RestoreV2:13*
T0*
_output_shapes
:
h
save/AssignVariableOp_8AssignVariableOptraining/Adam/dense_13/bias/vsave/Identity_8*
dtype0
Q
save/Identity_9Identitysave/RestoreV2:14*
T0*
_output_shapes
:
Z
save/AssignVariableOp_9AssignVariableOpdense_13/kernelsave/Identity_9*
dtype0
R
save/Identity_10Identitysave/RestoreV2:15*
T0*
_output_shapes
:
l
save/AssignVariableOp_10AssignVariableOptraining/Adam/dense_13/kernel/msave/Identity_10*
dtype0
R
save/Identity_11Identitysave/RestoreV2:16*
T0*
_output_shapes
:
l
save/AssignVariableOp_11AssignVariableOptraining/Adam/dense_13/kernel/vsave/Identity_11*
dtype0

save/NoOp_5NoOp
R
save/Identity_12Identitysave/RestoreV2:18*
T0*
_output_shapes
:
S
save/AssignVariableOp_12AssignVariableOpbeta_1save/Identity_12*
dtype0
R
save/Identity_13Identitysave/RestoreV2:19*
T0*
_output_shapes
:
S
save/AssignVariableOp_13AssignVariableOpbeta_2save/Identity_13*
dtype0
R
save/Identity_14Identitysave/RestoreV2:20*
T0*
_output_shapes
:
R
save/AssignVariableOp_14AssignVariableOpdecaysave/Identity_14*
dtype0
R
save/Identity_15Identitysave/RestoreV2:21*
T0*
_output_shapes
:
T
save/AssignVariableOp_15AssignVariableOpepsilonsave/Identity_15*
dtype0
a
save/Identity_16Identitysave/RestoreV2:22"/device:CPU:0*
T0	*
_output_shapes
:
`
save/AssignVariableOp_16AssignVariableOpitersave/Identity_16"/device:CPU:0*
dtype0	
R
save/Identity_17Identitysave/RestoreV2:23*
T0*
_output_shapes
:
Z
save/AssignVariableOp_17AssignVariableOplearning_ratesave/Identity_17*
dtype0
Ž
save/restore_all/NoOpNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_10^save/AssignVariableOp_11^save/AssignVariableOp_12^save/AssignVariableOp_13^save/AssignVariableOp_14^save/AssignVariableOp_15^save/AssignVariableOp_17^save/AssignVariableOp_2^save/AssignVariableOp_3^save/AssignVariableOp_4^save/AssignVariableOp_5^save/AssignVariableOp_6^save/AssignVariableOp_7^save/AssignVariableOp_8^save/AssignVariableOp_9
^save/NoOp^save/NoOp_1^save/NoOp_2^save/NoOp_3^save/NoOp_4^save/NoOp_5
I
save/restore_all/NoOp_1NoOp^save/AssignVariableOp_16"/device:CPU:0
J
save/restore_allNoOp^save/restore_all/NoOp^save/restore_all/NoOp_1
0
init_1NoOp^count_1/Assign^total_1/Assign"D
save/Const:0save/control_dependency:0save/restore_all 5 @F8"
trainable_variablesűř

dense_12/kernel:0dense_12/kernel/Assign%dense_12/kernel/Read/ReadVariableOp:0(2,dense_12/kernel/Initializer/random_uniform:08
s
dense_12/bias:0dense_12/bias/Assign#dense_12/bias/Read/ReadVariableOp:0(2!dense_12/bias/Initializer/zeros:08

dense_13/kernel:0dense_13/kernel/Assign%dense_13/kernel/Read/ReadVariableOp:0(2,dense_13/kernel/Initializer/random_uniform:08
s
dense_13/bias:0dense_13/bias/Assign#dense_13/bias/Read/ReadVariableOp:0(2!dense_13/bias/Initializer/zeros:08"Í
local_variablesšś
Y
	count_1:0count_1/Assigncount_1/Read/ReadVariableOp:0(2count_1/Initializer/zeros:0
Y
	total_1:0total_1/Assigntotal_1/Read/ReadVariableOp:0(2total_1/Initializer/zeros:0"Ă°
cond_contextą°­°

loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/cond_textloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_t:0 *	
sloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_scalar:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:1
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_t:0ů
sloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_scalar:0loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:1
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0
ł}
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/cond_text_1loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_f:0*;
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:1
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:1
źloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:0
źloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1
źloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:2
ľloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
ˇloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1
˛loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:0
Žloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:0
ˇloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
šloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1
´loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:0
°loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:0
Żloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:0
Şloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:0
´loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:0
łloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:0
łloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:0
­loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:0
Ľloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:0
Łloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
Śloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0
¨loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_f:0
uloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/values/rank:0
vloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/values/shape:0
vloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/weights/rank:0
wloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/weights/shape:0 
uloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/values/rank:0Śloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0ł
wloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/weights/shape:0ˇloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0°
vloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/values/shape:0ľloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0Ł
vloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/weights/rank:0¨loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:020
0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_textloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0 *ľ,
źloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:0
źloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1
źloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:2
ľloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
ˇloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1
˛loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:0
Žloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:0
ˇloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
šloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1
´loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:0
°loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:0
Żloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:0
Şloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:0
´loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:0
łloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:0
łloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:0
­loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:0
Ľloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:0
Łloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
vloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/values/shape:0
wloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/weights/shape:0ľ
wloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/weights/shape:0šloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1ô
ˇloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0ˇloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0˛
vloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/values/shape:0ˇloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1đ
ľloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0ľloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0ş
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:02ů
ö
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_text_1loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0*
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:1
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0ş
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0Ŕ
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0

}loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/cond_text}loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0~loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_t:0 *
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency:0
}loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0
~loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_t:0ţ
}loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0}loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0
Č
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/cond_text_1}loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0~loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f:0*Ĺ
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1:0
}loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0
~loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f:0
sloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_scalar:0
~loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge:0
vloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/values/shape:0
wloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/weights/shape:0ţ
}loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0}loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0
wloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/weights/shape:0loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1:0
vloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/values/shape:0loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2:0ý
sloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_scalar:0loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3:0
~loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge:0loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch:0"`
global_stepQO
M
iter:0iter/Assigniter/Read/ReadVariableOp:0(2iter/Initializer/zeros:0"
	variablesţ

dense_12/kernel:0dense_12/kernel/Assign%dense_12/kernel/Read/ReadVariableOp:0(2,dense_12/kernel/Initializer/random_uniform:08
s
dense_12/bias:0dense_12/bias/Assign#dense_12/bias/Read/ReadVariableOp:0(2!dense_12/bias/Initializer/zeros:08

dense_13/kernel:0dense_13/kernel/Assign%dense_13/kernel/Read/ReadVariableOp:0(2,dense_13/kernel/Initializer/random_uniform:08
s
dense_13/bias:0dense_13/bias/Assign#dense_13/bias/Read/ReadVariableOp:0(2!dense_13/bias/Initializer/zeros:08
M
iter:0iter/Assigniter/Read/ReadVariableOp:0(2iter/Initializer/zeros:0
y
learning_rate:0learning_rate/Assign#learning_rate/Read/ReadVariableOp:0(2)learning_rate/Initializer/initial_value:0
Y
decay:0decay/Assigndecay/Read/ReadVariableOp:0(2!decay/Initializer/initial_value:0
]
beta_1:0beta_1/Assignbeta_1/Read/ReadVariableOp:0(2"beta_1/Initializer/initial_value:0
]
beta_2:0beta_2/Assignbeta_2/Read/ReadVariableOp:0(2"beta_2/Initializer/initial_value:0
a
	epsilon:0epsilon/Assignepsilon/Read/ReadVariableOp:0(2#epsilon/Initializer/initial_value:0
š
!training/Adam/dense_12/kernel/m:0&training/Adam/dense_12/kernel/m/Assign5training/Adam/dense_12/kernel/m/Read/ReadVariableOp:0(23training/Adam/dense_12/kernel/m/Initializer/zeros:0
ą
training/Adam/dense_12/bias/m:0$training/Adam/dense_12/bias/m/Assign3training/Adam/dense_12/bias/m/Read/ReadVariableOp:0(21training/Adam/dense_12/bias/m/Initializer/zeros:0
š
!training/Adam/dense_13/kernel/m:0&training/Adam/dense_13/kernel/m/Assign5training/Adam/dense_13/kernel/m/Read/ReadVariableOp:0(23training/Adam/dense_13/kernel/m/Initializer/zeros:0
ą
training/Adam/dense_13/bias/m:0$training/Adam/dense_13/bias/m/Assign3training/Adam/dense_13/bias/m/Read/ReadVariableOp:0(21training/Adam/dense_13/bias/m/Initializer/zeros:0
š
!training/Adam/dense_12/kernel/v:0&training/Adam/dense_12/kernel/v/Assign5training/Adam/dense_12/kernel/v/Read/ReadVariableOp:0(23training/Adam/dense_12/kernel/v/Initializer/zeros:0
ą
training/Adam/dense_12/bias/v:0$training/Adam/dense_12/bias/v/Assign3training/Adam/dense_12/bias/v/Read/ReadVariableOp:0(21training/Adam/dense_12/bias/v/Initializer/zeros:0
š
!training/Adam/dense_13/kernel/v:0&training/Adam/dense_13/kernel/v/Assign5training/Adam/dense_13/kernel/v/Read/ReadVariableOp:0(23training/Adam/dense_13/kernel/v/Initializer/zeros:0
ą
training/Adam/dense_13/bias/v:0$training/Adam/dense_13/bias/v/Assign3training/Adam/dense_13/bias/v/Read/ReadVariableOp:0(21training/Adam/dense_13/bias/v/Initializer/zeros:0*Q
__saved_model_train_op75
__saved_model_train_op
training_1/group_deps*ó
trainé
:
dense_12_input(
dense_12_input:0˙˙˙˙˙˙˙˙˙
D
dense_13_target1
dense_13_target:0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
metrics/accuracy/update_op
metric_op_wrapper:0 -
metrics/accuracy/value
Identity_20:0 A
predictions/dense_13)
dense_13/Softmax:0˙˙˙˙˙˙˙˙˙

loss

loss/mul:0 tensorflow/supervised/training*@
__saved_model_init_op'%
__saved_model_init_op
init_1Śě
Ő
:
Add
x"T
y"T
z"T"
Ttype:
2	

ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeint
E
AssignAddVariableOp
resource
value"dtype"
dtypetype
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
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
š
DenseToDenseSetOperation	
set1"T	
set2"T
result_indices	
result_values"T
result_shape	"
set_operationstring"
validate_indicesbool("
Ttype:
	2	
5
DivNoNan
x"T
y"T
z"T"
Ttype:
2
B
Equal
x"T
y"T
z
"
Ttype:
2	

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
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

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2

#SparseSoftmaxCrossEntropyWithLogits
features"T
labels"Tlabels	
loss"T
backprop"T"
Ttype:
2"
Tlabelstype0	:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape
9
VarIsInitializedOp
resource
is_initialized
"eval*2.0.0-alpha02v1.12.0-9492-g2c319fb4158˝
s
dense_12_inputPlaceholder*
shape:˙˙˙˙˙˙˙˙˙*
dtype0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
0dense_12/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@dense_12/kernel*
valueB"     *
dtype0*
_output_shapes
:

.dense_12/kernel/Initializer/random_uniform/minConst*"
_class
loc:@dense_12/kernel*
valueB
 *HY˝*
dtype0*
_output_shapes
: 

.dense_12/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@dense_12/kernel*
valueB
 *HY=*
dtype0*
_output_shapes
: 
×
8dense_12/kernel/Initializer/random_uniform/RandomUniformRandomUniform0dense_12/kernel/Initializer/random_uniform/shape*
T0*"
_class
loc:@dense_12/kernel*
dtype0* 
_output_shapes
:

Ú
.dense_12/kernel/Initializer/random_uniform/subSub.dense_12/kernel/Initializer/random_uniform/max.dense_12/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_12/kernel*
_output_shapes
: 
î
.dense_12/kernel/Initializer/random_uniform/mulMul8dense_12/kernel/Initializer/random_uniform/RandomUniform.dense_12/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@dense_12/kernel* 
_output_shapes
:

ŕ
*dense_12/kernel/Initializer/random_uniformAdd.dense_12/kernel/Initializer/random_uniform/mul.dense_12/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_12/kernel* 
_output_shapes
:

 
dense_12/kernelVarHandleOp*
shape:
* 
shared_namedense_12/kernel*"
_class
loc:@dense_12/kernel*
dtype0*
_output_shapes
: 
o
0dense_12/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_12/kernel*
_output_shapes
: 

dense_12/kernel/AssignAssignVariableOpdense_12/kernel*dense_12/kernel/Initializer/random_uniform*"
_class
loc:@dense_12/kernel*
dtype0

#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*"
_class
loc:@dense_12/kernel*
dtype0* 
_output_shapes
:


dense_12/bias/Initializer/zerosConst* 
_class
loc:@dense_12/bias*
valueB*    *
dtype0*
_output_shapes	
:

dense_12/biasVarHandleOp*
shape:*
shared_namedense_12/bias* 
_class
loc:@dense_12/bias*
dtype0*
_output_shapes
: 
k
.dense_12/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_12/bias*
_output_shapes
: 

dense_12/bias/AssignAssignVariableOpdense_12/biasdense_12/bias/Initializer/zeros* 
_class
loc:@dense_12/bias*
dtype0

!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias* 
_class
loc:@dense_12/bias*
dtype0*
_output_shapes	
:
p
dense_12/MatMul/ReadVariableOpReadVariableOpdense_12/kernel*
dtype0* 
_output_shapes
:

|
dense_12/MatMulMatMuldense_12_inputdense_12/MatMul/ReadVariableOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
j
dense_12/BiasAdd/ReadVariableOpReadVariableOpdense_12/bias*
dtype0*
_output_shapes	
:

dense_12/BiasAddBiasAdddense_12/MatMuldense_12/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Z
dense_12/ReluReludense_12/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
dropout_6/IdentityIdentitydense_12/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
0dense_13/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@dense_13/kernel*
valueB"   
   *
dtype0*
_output_shapes
:

.dense_13/kernel/Initializer/random_uniform/minConst*"
_class
loc:@dense_13/kernel*
valueB
 *Ű˝*
dtype0*
_output_shapes
: 

.dense_13/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@dense_13/kernel*
valueB
 *Ű=*
dtype0*
_output_shapes
: 
Ö
8dense_13/kernel/Initializer/random_uniform/RandomUniformRandomUniform0dense_13/kernel/Initializer/random_uniform/shape*
T0*"
_class
loc:@dense_13/kernel*
dtype0*
_output_shapes
:	

Ú
.dense_13/kernel/Initializer/random_uniform/subSub.dense_13/kernel/Initializer/random_uniform/max.dense_13/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_13/kernel*
_output_shapes
: 
í
.dense_13/kernel/Initializer/random_uniform/mulMul8dense_13/kernel/Initializer/random_uniform/RandomUniform.dense_13/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@dense_13/kernel*
_output_shapes
:	

ß
*dense_13/kernel/Initializer/random_uniformAdd.dense_13/kernel/Initializer/random_uniform/mul.dense_13/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_13/kernel*
_output_shapes
:	


dense_13/kernelVarHandleOp*
shape:	
* 
shared_namedense_13/kernel*"
_class
loc:@dense_13/kernel*
dtype0*
_output_shapes
: 
o
0dense_13/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_13/kernel*
_output_shapes
: 

dense_13/kernel/AssignAssignVariableOpdense_13/kernel*dense_13/kernel/Initializer/random_uniform*"
_class
loc:@dense_13/kernel*
dtype0

#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*"
_class
loc:@dense_13/kernel*
dtype0*
_output_shapes
:	


dense_13/bias/Initializer/zerosConst* 
_class
loc:@dense_13/bias*
valueB
*    *
dtype0*
_output_shapes
:


dense_13/biasVarHandleOp*
shape:
*
shared_namedense_13/bias* 
_class
loc:@dense_13/bias*
dtype0*
_output_shapes
: 
k
.dense_13/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_13/bias*
_output_shapes
: 

dense_13/bias/AssignAssignVariableOpdense_13/biasdense_13/bias/Initializer/zeros* 
_class
loc:@dense_13/bias*
dtype0

!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias* 
_class
loc:@dense_13/bias*
dtype0*
_output_shapes
:

o
dense_13/MatMul/ReadVariableOpReadVariableOpdense_13/kernel*
dtype0*
_output_shapes
:	


dense_13/MatMulMatMuldropout_6/Identitydense_13/MatMul/ReadVariableOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

i
dense_13/BiasAdd/ReadVariableOpReadVariableOpdense_13/bias*
dtype0*
_output_shapes
:


dense_13/BiasAddBiasAdddense_13/MatMuldense_13/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

_
dense_13/SoftmaxSoftmaxdense_13/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙


dense_13_targetPlaceholder*%
shape:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
dtype0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
R
ConstConst*
valueB*  ?*
dtype0*
_output_shapes
:

dense_13_sample_weightsPlaceholderWithDefaultConst*
shape:˙˙˙˙˙˙˙˙˙*
dtype0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
v
total/Initializer/zerosConst*
_class

loc:@total*
valueB
 *    *
dtype0*
_output_shapes
: 
x
totalVarHandleOp*
shape: *
shared_nametotal*
_class

loc:@total*
dtype0*
_output_shapes
: 
[
&total/IsInitialized/VarIsInitializedOpVarIsInitializedOptotal*
_output_shapes
: 
g
total/AssignAssignVariableOptotaltotal/Initializer/zeros*
_class

loc:@total*
dtype0
q
total/Read/ReadVariableOpReadVariableOptotal*
_class

loc:@total*
dtype0*
_output_shapes
: 
v
count/Initializer/zerosConst*
_class

loc:@count*
valueB
 *    *
dtype0*
_output_shapes
: 
x
countVarHandleOp*
shape: *
shared_namecount*
_class

loc:@count*
dtype0*
_output_shapes
: 
[
&count/IsInitialized/VarIsInitializedOpVarIsInitializedOpcount*
_output_shapes
: 
g
count/AssignAssignVariableOpcountcount/Initializer/zeros*
_class

loc:@count*
dtype0
q
count/Read/ReadVariableOpReadVariableOpcount*
_class

loc:@count*
dtype0*
_output_shapes
: 

metrics/accuracy/SqueezeSqueezedense_13_target*
squeeze_dims

˙˙˙˙˙˙˙˙˙*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
l
!metrics/accuracy/ArgMax/dimensionConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 

metrics/accuracy/ArgMaxArgMaxdense_13/Softmax!metrics/accuracy/ArgMax/dimension*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
metrics/accuracy/CastCastmetrics/accuracy/ArgMax*

SrcT0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0
~
metrics/accuracy/EqualEqualmetrics/accuracy/Squeezemetrics/accuracy/Cast*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
metrics/accuracy/Cast_1Castmetrics/accuracy/Equal*

SrcT0
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0
`
metrics/accuracy/ConstConst*
valueB: *
dtype0*
_output_shapes
:
m
metrics/accuracy/SumSummetrics/accuracy/Cast_1metrics/accuracy/Const*
T0*
_output_shapes
: 
e
$metrics/accuracy/AssignAddVariableOpAssignAddVariableOptotalmetrics/accuracy/Sum*
dtype0

metrics/accuracy/ReadVariableOpReadVariableOptotal%^metrics/accuracy/AssignAddVariableOp^metrics/accuracy/Sum*
dtype0*
_output_shapes
: 
W
metrics/accuracy/SizeSizemetrics/accuracy/Cast_1*
T0*
_output_shapes
: 
f
metrics/accuracy/Cast_2Castmetrics/accuracy/Size*

SrcT0*
_output_shapes
: *

DstT0

&metrics/accuracy/AssignAddVariableOp_1AssignAddVariableOpcountmetrics/accuracy/Cast_2%^metrics/accuracy/AssignAddVariableOp*
dtype0
Ż
!metrics/accuracy/ReadVariableOp_1ReadVariableOpcount%^metrics/accuracy/AssignAddVariableOp'^metrics/accuracy/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 

*metrics/accuracy/div_no_nan/ReadVariableOpReadVariableOptotal'^metrics/accuracy/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 

,metrics/accuracy/div_no_nan/ReadVariableOp_1ReadVariableOpcount'^metrics/accuracy/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
˘
metrics/accuracy/div_no_nanDivNoNan*metrics/accuracy/div_no_nan/ReadVariableOp,metrics/accuracy/div_no_nan/ReadVariableOp_1*
T0*
_output_shapes
: 
c
metrics/accuracy/IdentityIdentitymetrics/accuracy/div_no_nan*
T0*
_output_shapes
: 

metrics/accuracy/Squeeze_1Squeezedense_13_target*
squeeze_dims

˙˙˙˙˙˙˙˙˙*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
n
#metrics/accuracy/ArgMax_1/dimensionConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 

metrics/accuracy/ArgMax_1ArgMaxdense_13/Softmax#metrics/accuracy/ArgMax_1/dimension*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
w
metrics/accuracy/Cast_3Castmetrics/accuracy/ArgMax_1*

SrcT0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0

metrics/accuracy/Equal_1Equalmetrics/accuracy/Squeeze_1metrics/accuracy/Cast_3*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
v
metrics/accuracy/Cast_4Castmetrics/accuracy/Equal_1*

SrcT0
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0
b
metrics/accuracy/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
q
metrics/accuracy/MeanMeanmetrics/accuracy/Cast_4metrics/accuracy/Const_1*
T0*
_output_shapes
: 

@loss/dense_13_loss/sparse_categorical_crossentropy/Reshape/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
Ć
:loss/dense_13_loss/sparse_categorical_crossentropy/ReshapeReshapedense_13_target@loss/dense_13_loss/sparse_categorical_crossentropy/Reshape/shape*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
¸
7loss/dense_13_loss/sparse_categorical_crossentropy/CastCast:loss/dense_13_loss/sparse_categorical_crossentropy/Reshape*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0	

Bloss/dense_13_loss/sparse_categorical_crossentropy/Reshape_1/shapeConst*
valueB"˙˙˙˙
   *
dtype0*
_output_shapes
:
Ď
<loss/dense_13_loss/sparse_categorical_crossentropy/Reshape_1Reshapedense_13/BiasAddBloss/dense_13_loss/sparse_categorical_crossentropy/Reshape_1/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ă
\loss/dense_13_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/ShapeShape7loss/dense_13_loss/sparse_categorical_crossentropy/Cast*
T0	*
_output_shapes
:
Ů
zloss/dense_13_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits<loss/dense_13_loss/sparse_categorical_crossentropy/Reshape_17loss/dense_13_loss/sparse_categorical_crossentropy/Cast*
T0*6
_output_shapes$
":˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

ź
uloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/weights/shapeShapedense_13_sample_weights*
T0*
_output_shapes
:
ś
tloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B :*
dtype0*
_output_shapes
: 

tloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/values/shapeShapezloss/dense_13_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*
_output_shapes
:
ľ
sloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
ľ
sloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_scalar/xConst*
value	B : *
dtype0*
_output_shapes
: 

qloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_scalarEqualsloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_scalar/xtloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/weights/rank*
T0*
_output_shapes
: 

}loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/SwitchSwitchqloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_scalarqloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
­
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_tIdentityloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch:1*
T0
*
_output_shapes
: 
Ť
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_fIdentity}loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch*
T0
*
_output_shapes
: 

~loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_idIdentityqloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: 
Ś
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1Switchqloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_scalar~loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0
*
_classz
xvloc:@loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_scalar*
_output_shapes
: : 

loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqual¤loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchŚloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
T0*
_output_shapes
: 
Đ
¤loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitchsloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/values/rank~loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0*
_class|
zxloc:@loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/values/rank*
_output_shapes
: : 
Ô
Śloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1Switchtloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/weights/rank~loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0*
_class}
{yloc:@loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/weights/rank*
_output_shapes
: : 

loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitchloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: : 
ă
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentityloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
T0
*
_output_shapes
: 
á
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentityloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
T0
*
_output_shapes
: 
ć
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentityloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: 

°loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConst^loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
Ń
Źloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDimsˇloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1°loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*
T0*
_output_shapes

:
é
łloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitchtloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/values/shape~loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0*
_class}
{yloc:@loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/values/shape* 
_output_shapes
::
Ć
ľloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1Switchłloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switchloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*
_class}
{yloc:@loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/values/shape* 
_output_shapes
::
 
ąloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConst^loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB"      *
dtype0*
_output_shapes
:

ąloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConst^loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
Ĺ
Ťloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFilląloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shapeąloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
T0*
_output_shapes

:

­loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConst^loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
ô
¨loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2Źloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDimsŤloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like­loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*
T0*
N*
_output_shapes

:

˛loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConst^loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
×
Žloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDimsšloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1˛loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*
T0*
_output_shapes

:
í
ľloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitchuloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/weights/shape~loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0*
_class~
|zloc:@loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/weights/shape* 
_output_shapes
::
Ë
ˇloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1Switchľloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switchloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*
_class~
|zloc:@loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/weights/shape* 
_output_shapes
::

şloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperationŽloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1¨loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*
set_operationa-b*
T0*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:

˛loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSizeźloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
T0*
_output_shapes
: 

Łloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConst^loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B : *
dtype0*
_output_shapes
: 
§
Ąloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqualŁloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x˛loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
T0*
_output_shapes
: 
¸
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Switchloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0
*ł
_class¨
Ľ˘loc:@loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: : 

loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMergeloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Ąloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
T0
*
N*
_output_shapes
: : 
Î
|loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/MergeMergeloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Mergeloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:1*
T0
*
N*
_output_shapes
: : 
Ő
mloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/ConstConst*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
ž
oloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/Const_1Const*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
É
oloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/Const_2Const**
value!B Bdense_13_sample_weights:0*
dtype0*
_output_shapes
: 
˝
oloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/Const_3Const*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
Ż
oloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/Const_4Const*
valueB B|loss/dense_13_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:0*
dtype0*
_output_shapes
: 
ş
oloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/Const_5Const*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
Ł
zloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/SwitchSwitch|loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge|loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
§
|loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_tIdentity|loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
Ľ
|loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_fIdentityzloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Switch*
T0
*
_output_shapes
: 
Ś
{loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_idIdentity|loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: 
˙
xloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/NoOpNoOp}^loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_t
Â
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependencyIdentity|loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_ty^loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/NoOp*
T0
*
_class
loc:@loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_t*
_output_shapes
: 
é
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0Const}^loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
Đ
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1Const}^loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
Ű
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2Const}^loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f**
value!B Bdense_13_sample_weights:0*
dtype0*
_output_shapes
: 
Ď
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4Const}^loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
Á
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5Const}^loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
valueB B|loss/dense_13_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:0*
dtype0*
_output_shapes
: 
Ě
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7Const}^loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
Ć
zloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/AssertAssertloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switchloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3*
T
2	

ż
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/SwitchSwitch|loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge{loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*
T0
*
_class
loc:@loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: : 
¸
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1Switchuloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/weights/shape{loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*
T0*
_class~
|zloc:@loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/weights/shape* 
_output_shapes
::
ś
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2Switchtloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/values/shape{loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*
T0*
_class}
{yloc:@loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/values/shape* 
_output_shapes
::
¨
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3Switchqloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_scalar{loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*
T0
*
_classz
xvloc:@loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_scalar*
_output_shapes
: : 
Ć
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1Identity|loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f{^loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert*
T0
*
_class
loc:@loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: 
Â
yloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/MergeMergeloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 

bloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/ones_like/ShapeShapezloss/dense_13_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsz^loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Merge*
T0*
_output_shapes
:
Ł
bloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/ones_like/ConstConstz^loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Merge*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ú
\loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/ones_likeFillbloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/ones_like/Shapebloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/ones_like/Const*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ţ
Rloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weightsMuldense_13_sample_weights\loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/ones_like*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
É
Dloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/MulMulzloss/dense_13_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsRloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
loss/dense_13_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:

loss/dense_13_loss/SumSumDloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/Mulloss/dense_13_loss/Const*
T0*
_output_shapes
: 

loss/dense_13_loss/num_elementsSizeDloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/Mul*
T0*
_output_shapes
: 
}
$loss/dense_13_loss/num_elements/CastCastloss/dense_13_loss/num_elements*

SrcT0*
_output_shapes
: *

DstT0
]
loss/dense_13_loss/mul/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
~
loss/dense_13_loss/mulMulloss/dense_13_loss/mul/x$loss/dense_13_loss/num_elements/Cast*
T0*
_output_shapes
: 
]
loss/dense_13_loss/Const_1Const*
valueB *
dtype0*
_output_shapes
: 
t
loss/dense_13_loss/Sum_1Sumloss/dense_13_loss/Sumloss/dense_13_loss/Const_1*
T0*
_output_shapes
: 
w
loss/dense_13_loss/valueDivNoNanloss/dense_13_loss/Sum_1loss/dense_13_loss/mul*
T0*
_output_shapes
: 
O

loss/mul/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
V
loss/mulMul
loss/mul/xloss/dense_13_loss/value*
T0*
_output_shapes
: 
q
iter/Initializer/zerosConst*
_class
	loc:@iter*
value	B	 R *
dtype0	*
_output_shapes
: 

iterVarHandleOp"/device:CPU:0*
shape: *
shared_nameiter*
_class
	loc:@iter*
dtype0	*
_output_shapes
: 
h
%iter/IsInitialized/VarIsInitializedOpVarIsInitializedOpiter"/device:CPU:0*
_output_shapes
: 
r
iter/AssignAssignVariableOpiteriter/Initializer/zeros"/device:CPU:0*
_class
	loc:@iter*
dtype0	
}
iter/Read/ReadVariableOpReadVariableOpiter"/device:CPU:0*
_class
	loc:@iter*
dtype0	*
_output_shapes
: 

'learning_rate/Initializer/initial_valueConst* 
_class
loc:@learning_rate*
valueB
 *o:*
dtype0*
_output_shapes
: 

learning_rateVarHandleOp*
shape: *
shared_namelearning_rate* 
_class
loc:@learning_rate*
dtype0*
_output_shapes
: 
k
.learning_rate/IsInitialized/VarIsInitializedOpVarIsInitializedOplearning_rate*
_output_shapes
: 

learning_rate/AssignAssignVariableOplearning_rate'learning_rate/Initializer/initial_value* 
_class
loc:@learning_rate*
dtype0

!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate* 
_class
loc:@learning_rate*
dtype0*
_output_shapes
: 
~
decay/Initializer/initial_valueConst*
_class

loc:@decay*
valueB
 *    *
dtype0*
_output_shapes
: 
x
decayVarHandleOp*
shape: *
shared_namedecay*
_class

loc:@decay*
dtype0*
_output_shapes
: 
[
&decay/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecay*
_output_shapes
: 
o
decay/AssignAssignVariableOpdecaydecay/Initializer/initial_value*
_class

loc:@decay*
dtype0
q
decay/Read/ReadVariableOpReadVariableOpdecay*
_class

loc:@decay*
dtype0*
_output_shapes
: 

 beta_1/Initializer/initial_valueConst*
_class
loc:@beta_1*
valueB
 *fff?*
dtype0*
_output_shapes
: 
{
beta_1VarHandleOp*
shape: *
shared_namebeta_1*
_class
loc:@beta_1*
dtype0*
_output_shapes
: 
]
'beta_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpbeta_1*
_output_shapes
: 
s
beta_1/AssignAssignVariableOpbeta_1 beta_1/Initializer/initial_value*
_class
loc:@beta_1*
dtype0
t
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_class
loc:@beta_1*
dtype0*
_output_shapes
: 

 beta_2/Initializer/initial_valueConst*
_class
loc:@beta_2*
valueB
 *wž?*
dtype0*
_output_shapes
: 
{
beta_2VarHandleOp*
shape: *
shared_namebeta_2*
_class
loc:@beta_2*
dtype0*
_output_shapes
: 
]
'beta_2/IsInitialized/VarIsInitializedOpVarIsInitializedOpbeta_2*
_output_shapes
: 
s
beta_2/AssignAssignVariableOpbeta_2 beta_2/Initializer/initial_value*
_class
loc:@beta_2*
dtype0
t
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_class
loc:@beta_2*
dtype0*
_output_shapes
: 

!epsilon/Initializer/initial_valueConst*
_class
loc:@epsilon*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
~
epsilonVarHandleOp*
shape: *
shared_name	epsilon*
_class
loc:@epsilon*
dtype0*
_output_shapes
: 
_
(epsilon/IsInitialized/VarIsInitializedOpVarIsInitializedOpepsilon*
_output_shapes
: 
w
epsilon/AssignAssignVariableOpepsilon!epsilon/Initializer/initial_value*
_class
loc:@epsilon*
dtype0
w
epsilon/Read/ReadVariableOpReadVariableOpepsilon*
_class
loc:@epsilon*
dtype0*
_output_shapes
: 
@
evaluation/group_depsNoOp	^loss/mul^metrics/accuracy/Mean
W
Const_1Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_2Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_3Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_4Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_5Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
Q
VarIsInitializedOpVarIsInitializedOpdense_13/kernel*
_output_shapes
: 
S
VarIsInitializedOp_1VarIsInitializedOpdense_12/kernel*
_output_shapes
: 
Q
VarIsInitializedOp_2VarIsInitializedOpdense_12/bias*
_output_shapes
: 
I
VarIsInitializedOp_3VarIsInitializedOpdecay*
_output_shapes
: 
Q
VarIsInitializedOp_4VarIsInitializedOplearning_rate*
_output_shapes
: 
J
VarIsInitializedOp_5VarIsInitializedOpbeta_1*
_output_shapes
: 
I
VarIsInitializedOp_6VarIsInitializedOptotal*
_output_shapes
: 
K
VarIsInitializedOp_7VarIsInitializedOpepsilon*
_output_shapes
: 
H
VarIsInitializedOp_8VarIsInitializedOpiter*
_output_shapes
: 
Q
VarIsInitializedOp_9VarIsInitializedOpdense_13/bias*
_output_shapes
: 
K
VarIsInitializedOp_10VarIsInitializedOpbeta_2*
_output_shapes
: 
J
VarIsInitializedOp_11VarIsInitializedOpcount*
_output_shapes
: 
ć
	init/NoOpNoOp^beta_1/Assign^beta_2/Assign^count/Assign^decay/Assign^dense_12/bias/Assign^dense_12/kernel/Assign^dense_13/bias/Assign^dense_13/kernel/Assign^epsilon/Assign^learning_rate/Assign^total/Assign
0
init/NoOp_1NoOp^iter/Assign"/device:CPU:0
&
initNoOp
^init/NoOp^init/NoOp_1
W
Const_6Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
\
Const_7Const"/device:CPU:0*
valueB Bmodel*
dtype0*
_output_shapes
: 
W
Const_8Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_9Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_10Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_11Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_12Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_13Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
¤
RestoreV2/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
r
RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

	RestoreV2	RestoreV2Const_7RestoreV2/tensor_namesRestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
B
IdentityIdentity	RestoreV2*
T0*
_output_shapes
:
L
AssignVariableOpAssignVariableOpdense_12/kernelIdentity*
dtype0
¤
RestoreV2_1/tensor_namesConst"/device:CPU:0*I
value@B>B4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_1	RestoreV2Const_7RestoreV2_1/tensor_namesRestoreV2_1/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
F

Identity_1IdentityRestoreV2_1*
T0*
_output_shapes
:
N
AssignVariableOp_1AssignVariableOpdense_12/bias
Identity_1*
dtype0
Ś
RestoreV2_2/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_2/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_2	RestoreV2Const_7RestoreV2_2/tensor_namesRestoreV2_2/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
F

Identity_2IdentityRestoreV2_2*
T0*
_output_shapes
:
P
AssignVariableOp_2AssignVariableOpdense_13/kernel
Identity_2*
dtype0
¤
RestoreV2_3/tensor_namesConst"/device:CPU:0*I
value@B>B4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_3/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_3	RestoreV2Const_7RestoreV2_3/tensor_namesRestoreV2_3/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
F

Identity_3IdentityRestoreV2_3*
T0*
_output_shapes
:
N
AssignVariableOp_3AssignVariableOpdense_13/bias
Identity_3*
dtype0

RestoreV2_4/tensor_namesConst"/device:CPU:0*>
value5B3B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_4/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_4	RestoreV2Const_7RestoreV2_4/tensor_namesRestoreV2_4/shape_and_slices"/device:CPU:0*
dtypes
2	*
_output_shapes
:
U

Identity_4IdentityRestoreV2_4"/device:CPU:0*
T0	*
_output_shapes
:
T
AssignVariableOp_4AssignVariableOpiter
Identity_4"/device:CPU:0*
dtype0	
˘
RestoreV2_5/tensor_namesConst"/device:CPU:0*G
value>B<B2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_5/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_5	RestoreV2Const_7RestoreV2_5/tensor_namesRestoreV2_5/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
F

Identity_5IdentityRestoreV2_5*
T0*
_output_shapes
:
N
AssignVariableOp_5AssignVariableOplearning_rate
Identity_5*
dtype0

RestoreV2_6/tensor_namesConst"/device:CPU:0*?
value6B4B*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_6/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_6	RestoreV2Const_7RestoreV2_6/tensor_namesRestoreV2_6/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
F

Identity_6IdentityRestoreV2_6*
T0*
_output_shapes
:
F
AssignVariableOp_6AssignVariableOpdecay
Identity_6*
dtype0

RestoreV2_7/tensor_namesConst"/device:CPU:0*@
value7B5B+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_7/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_7	RestoreV2Const_7RestoreV2_7/tensor_namesRestoreV2_7/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
F

Identity_7IdentityRestoreV2_7*
T0*
_output_shapes
:
G
AssignVariableOp_7AssignVariableOpbeta_1
Identity_7*
dtype0

RestoreV2_8/tensor_namesConst"/device:CPU:0*@
value7B5B+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_8/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_8	RestoreV2Const_7RestoreV2_8/tensor_namesRestoreV2_8/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
F

Identity_8IdentityRestoreV2_8*
T0*
_output_shapes
:
G
AssignVariableOp_8AssignVariableOpbeta_2
Identity_8*
dtype0

RestoreV2_9/tensor_namesConst"/device:CPU:0*A
value8B6B,optimizer/epsilon/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_9/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_9	RestoreV2Const_7RestoreV2_9/tensor_namesRestoreV2_9/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
F

Identity_9IdentityRestoreV2_9*
T0*
_output_shapes
:
H
AssignVariableOp_9AssignVariableOpepsilon
Identity_9*
dtype0
z
total_1/Initializer/zerosConst*
_class
loc:@total_1*
valueB
 *    *
dtype0*
_output_shapes
: 
~
total_1VarHandleOp*
shape: *
shared_name	total_1*
_class
loc:@total_1*
dtype0*
_output_shapes
: 
_
(total_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptotal_1*
_output_shapes
: 
o
total_1/AssignAssignVariableOptotal_1total_1/Initializer/zeros*
_class
loc:@total_1*
dtype0
w
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_class
loc:@total_1*
dtype0*
_output_shapes
: 
z
count_1/Initializer/zerosConst*
_class
loc:@count_1*
valueB
 *    *
dtype0*
_output_shapes
: 
~
count_1VarHandleOp*
shape: *
shared_name	count_1*
_class
loc:@count_1*
dtype0*
_output_shapes
: 
_
(count_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpcount_1*
_output_shapes
: 
o
count_1/AssignAssignVariableOpcount_1count_1/Initializer/zeros*
_class
loc:@count_1*
dtype0
w
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_class
loc:@count_1*
dtype0*
_output_shapes
: 
K
Const_14Const*
valueB *
dtype0*
_output_shapes
: 
L
SumSummetrics/accuracy/MeanConst_14*
T0*
_output_shapes
: 
E
AssignAddVariableOpAssignAddVariableOptotal_1Sum*
dtype0
j
ReadVariableOpReadVariableOptotal_1^AssignAddVariableOp^Sum*
dtype0*
_output_shapes
: 
F
SizeConst*
value	B :*
dtype0*
_output_shapes
: 
B
CastCastSize*

SrcT0*
_output_shapes
: *

DstT0
^
AssignAddVariableOp_1AssignAddVariableOpcount_1Cast^AssignAddVariableOp*
dtype0
~
ReadVariableOp_1ReadVariableOpcount_1^AssignAddVariableOp^AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
q
div_no_nan/ReadVariableOpReadVariableOptotal_1^AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
s
div_no_nan/ReadVariableOp_1ReadVariableOpcount_1^AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
o

div_no_nanDivNoNandiv_no_nan/ReadVariableOpdiv_no_nan/ReadVariableOp_1*
T0*
_output_shapes
: 
D
Identity_10Identity
div_no_nan*
T0*
_output_shapes
: 
[
div_no_nan_1/ReadVariableOpReadVariableOptotal_1*
dtype0*
_output_shapes
: 
]
div_no_nan_1/ReadVariableOp_1ReadVariableOpcount_1*
dtype0*
_output_shapes
: 
u
div_no_nan_1DivNoNandiv_no_nan_1/ReadVariableOpdiv_no_nan_1/ReadVariableOp_1*
T0*
_output_shapes
: 
F
Identity_11Identitydiv_no_nan_1*
T0*
_output_shapes
: 
l
metric_op_wrapperConst^AssignAddVariableOp_1*
valueB *
dtype0*
_output_shapes
: 
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
dtype0*
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
dtype0*
_output_shapes
: 
¨	
save/Const_1Const*ë
valueáBŢ B×{"class_name": "Sequential", "config": {"layers": [{"class_name": "Dense", "config": {"activation": "relu", "activity_regularizer": null, "batch_input_shape": [null, 784], "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "dtype": "float32", "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "kernel_regularizer": null, "name": "dense_12", "trainable": true, "units": 512, "use_bias": true}}, {"class_name": "Dropout", "config": {"dtype": "float32", "name": "dropout_6", "noise_shape": null, "rate": 0.2, "seed": null, "trainable": true}}, {"class_name": "Dense", "config": {"activation": "softmax", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "dtype": "float32", "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "kernel_regularizer": null, "name": "dense_13", "trainable": true, "units": 10, "use_bias": true}}], "name": "sequential_6"}}*
dtype0*
_output_shapes
: 
Ú
save/Const_2Const*
valueB B{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 784], "dtype": "float32", "name": "dense_12_input", "sparse": false}}*
dtype0*
_output_shapes
: 
â
save/Const_3Const*Ľ
valueB B{"class_name": "Dropout", "config": {"dtype": "float32", "name": "dropout_6", "noise_shape": null, "rate": 0.2, "seed": null, "trainable": true}}*
dtype0*
_output_shapes
: 

save/Const_4Const*Ţ
valueÔBŃ BĘ{"class_name": "Dense", "config": {"activation": "relu", "activity_regularizer": null, "batch_input_shape": [null, 784], "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "dtype": "float32", "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "kernel_regularizer": null, "name": "dense_12", "trainable": true, "units": 512, "use_bias": true}}*
dtype0*
_output_shapes
: 
ű
save/Const_5Const*ž
value´Bą BŞ{"class_name": "Dense", "config": {"activation": "softmax", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "dtype": "float32", "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "kernel_regularizer": null, "name": "dense_13", "trainable": true, "units": 10, "use_bias": true}}*
dtype0*
_output_shapes
: 
N
save/VarIsInitializedOpVarIsInitializedOptotal_1*
_output_shapes
: 
P
save/VarIsInitializedOp_1VarIsInitializedOpcount_1*
_output_shapes
: 
3
	save/initNoOp^count_1/Assign^total_1/Assign
Ş
save/Const_6Const*í
valueăBŕ BŮ{"class_name": "Adam", "config": {"amsgrad": false, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "decay": 0.0, "epsilon": 1.0000000116860974e-07, "learning_rate": 0.0010000000474974513, "name": "Adam"}}*
dtype0*
_output_shapes
: 
Ö
save/SaveV2/tensor_namesConst*
value˙BüB/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-2/.ATTRIBUTES/OBJECT_CONFIG_JSONB3layer_with_weights-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-1/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/.ATTRIBUTES/OBJECT_CONFIG_JSONB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB,optimizer/epsilon/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:

save/SaveV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:

save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicessave/Const_1save/Const_2save/Const_3save/Const_4!dense_12/bias/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOpsave/Const_5!dense_13/bias/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOpsave/Const_6beta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOpepsilon/Read/ReadVariableOpiter/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp*
dtypes
2	
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
č
save/RestoreV2/tensor_namesConst"/device:CPU:0*
value˙BüB/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-2/.ATTRIBUTES/OBJECT_CONFIG_JSONB3layer_with_weights-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-1/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/.ATTRIBUTES/OBJECT_CONFIG_JSONB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB,optimizer/epsilon/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:

save/RestoreV2/shape_and_slicesConst"/device:CPU:0*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ę
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2	*T
_output_shapesB
@::::::::::::::::

	save/NoOpNoOp

save/NoOp_1NoOp

save/NoOp_2NoOp

save/NoOp_3NoOp
N
save/IdentityIdentitysave/RestoreV2:4*
T0*
_output_shapes
:
T
save/AssignVariableOpAssignVariableOpdense_12/biassave/Identity*
dtype0
P
save/Identity_1Identitysave/RestoreV2:5*
T0*
_output_shapes
:
Z
save/AssignVariableOp_1AssignVariableOpdense_12/kernelsave/Identity_1*
dtype0

save/NoOp_4NoOp
P
save/Identity_2Identitysave/RestoreV2:7*
T0*
_output_shapes
:
X
save/AssignVariableOp_2AssignVariableOpdense_13/biassave/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:8*
T0*
_output_shapes
:
Z
save/AssignVariableOp_3AssignVariableOpdense_13/kernelsave/Identity_3*
dtype0

save/NoOp_5NoOp
Q
save/Identity_4Identitysave/RestoreV2:10*
T0*
_output_shapes
:
Q
save/AssignVariableOp_4AssignVariableOpbeta_1save/Identity_4*
dtype0
Q
save/Identity_5Identitysave/RestoreV2:11*
T0*
_output_shapes
:
Q
save/AssignVariableOp_5AssignVariableOpbeta_2save/Identity_5*
dtype0
Q
save/Identity_6Identitysave/RestoreV2:12*
T0*
_output_shapes
:
P
save/AssignVariableOp_6AssignVariableOpdecaysave/Identity_6*
dtype0
Q
save/Identity_7Identitysave/RestoreV2:13*
T0*
_output_shapes
:
R
save/AssignVariableOp_7AssignVariableOpepsilonsave/Identity_7*
dtype0
`
save/Identity_8Identitysave/RestoreV2:14"/device:CPU:0*
T0	*
_output_shapes
:
^
save/AssignVariableOp_8AssignVariableOpitersave/Identity_8"/device:CPU:0*
dtype0	
Q
save/Identity_9Identitysave/RestoreV2:15*
T0*
_output_shapes
:
X
save/AssignVariableOp_9AssignVariableOplearning_ratesave/Identity_9*
dtype0
×
save/restore_all/NoOpNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_2^save/AssignVariableOp_3^save/AssignVariableOp_4^save/AssignVariableOp_5^save/AssignVariableOp_6^save/AssignVariableOp_7^save/AssignVariableOp_9
^save/NoOp^save/NoOp_1^save/NoOp_2^save/NoOp_3^save/NoOp_4^save/NoOp_5
H
save/restore_all/NoOp_1NoOp^save/AssignVariableOp_8"/device:CPU:0
J
save/restore_allNoOp^save/restore_all/NoOp^save/restore_all/NoOp_1
0
init_1NoOp^count_1/Assign^total_1/Assign"D
save/Const:0save/control_dependency:0save/restore_all 5 @F8"
trainable_variablesűř

dense_12/kernel:0dense_12/kernel/Assign%dense_12/kernel/Read/ReadVariableOp:0(2,dense_12/kernel/Initializer/random_uniform:08
s
dense_12/bias:0dense_12/bias/Assign#dense_12/bias/Read/ReadVariableOp:0(2!dense_12/bias/Initializer/zeros:08

dense_13/kernel:0dense_13/kernel/Assign%dense_13/kernel/Read/ReadVariableOp:0(2,dense_13/kernel/Initializer/random_uniform:08
s
dense_13/bias:0dense_13/bias/Assign#dense_13/bias/Read/ReadVariableOp:0(2!dense_13/bias/Initializer/zeros:08"Í
local_variablesšś
Y
	count_1:0count_1/Assigncount_1/Read/ReadVariableOp:0(2count_1/Initializer/zeros:0
Y
	total_1:0total_1/Assigntotal_1/Read/ReadVariableOp:0(2total_1/Initializer/zeros:0"Ă°
cond_contextą°­°

loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/cond_textloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_t:0 *	
sloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_scalar:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:1
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_t:0ů
sloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_scalar:0loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:1
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0
ł}
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/cond_text_1loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_f:0*;
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:1
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:1
źloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:0
źloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1
źloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:2
ľloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
ˇloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1
˛loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:0
Žloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:0
ˇloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
šloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1
´loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:0
°loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:0
Żloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:0
Şloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:0
´loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:0
łloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:0
łloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:0
­loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:0
Ľloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:0
Łloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
Śloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0
¨loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_f:0
uloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/values/rank:0
vloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/values/shape:0
vloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/weights/rank:0
wloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/weights/shape:0 
uloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/values/rank:0Śloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0ł
wloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/weights/shape:0ˇloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0°
vloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/values/shape:0ľloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0Ł
vloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/weights/rank:0¨loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:020
0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_textloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0 *ľ,
źloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:0
źloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1
źloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:2
ľloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
ˇloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1
˛loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:0
Žloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:0
ˇloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
šloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1
´loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:0
°loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:0
Żloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:0
Şloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:0
´loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:0
łloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:0
łloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:0
­loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:0
Ľloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:0
Łloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
vloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/values/shape:0
wloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/weights/shape:0ľ
wloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/weights/shape:0šloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1ô
ˇloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0ˇloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0˛
vloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/values/shape:0ˇloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1đ
ľloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0ľloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0ş
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:02ů
ö
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_text_1loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0*
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:1
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0ş
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0Ŕ
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0

}loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/cond_text}loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0~loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_t:0 *
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency:0
}loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0
~loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_t:0ţ
}loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0}loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0
Č
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/cond_text_1}loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0~loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f:0*Ĺ
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7:0
loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1:0
}loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0
~loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f:0
sloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_scalar:0
~loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge:0
vloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/values/shape:0
wloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/weights/shape:0ţ
}loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0}loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0
wloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/weights/shape:0loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1:0
vloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/values/shape:0loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2:0ý
sloss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_scalar:0loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3:0
~loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge:0loss/dense_13_loss/sparse_categorical_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch:0"`
global_stepQO
M
iter:0iter/Assigniter/Read/ReadVariableOp:0(2iter/Initializer/zeros:0"Ď
	variablesÁž

dense_12/kernel:0dense_12/kernel/Assign%dense_12/kernel/Read/ReadVariableOp:0(2,dense_12/kernel/Initializer/random_uniform:08
s
dense_12/bias:0dense_12/bias/Assign#dense_12/bias/Read/ReadVariableOp:0(2!dense_12/bias/Initializer/zeros:08

dense_13/kernel:0dense_13/kernel/Assign%dense_13/kernel/Read/ReadVariableOp:0(2,dense_13/kernel/Initializer/random_uniform:08
s
dense_13/bias:0dense_13/bias/Assign#dense_13/bias/Read/ReadVariableOp:0(2!dense_13/bias/Initializer/zeros:08
M
iter:0iter/Assigniter/Read/ReadVariableOp:0(2iter/Initializer/zeros:0
y
learning_rate:0learning_rate/Assign#learning_rate/Read/ReadVariableOp:0(2)learning_rate/Initializer/initial_value:0
Y
decay:0decay/Assigndecay/Read/ReadVariableOp:0(2!decay/Initializer/initial_value:0
]
beta_1:0beta_1/Assignbeta_1/Read/ReadVariableOp:0(2"beta_1/Initializer/initial_value:0
]
beta_2:0beta_2/Assignbeta_2/Read/ReadVariableOp:0(2"beta_2/Initializer/initial_value:0
a
	epsilon:0epsilon/Assignepsilon/Read/ReadVariableOp:0(2#epsilon/Initializer/initial_value:0*î
evalĺ
:
dense_12_input(
dense_12_input:0˙˙˙˙˙˙˙˙˙
D
dense_13_target1
dense_13_target:0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙9
metrics/accuracy/update_op
metric_op_wrapper:0 -
metrics/accuracy/value
Identity_11:0 A
predictions/dense_13)
dense_13/Softmax:0˙˙˙˙˙˙˙˙˙

loss

loss/mul:0 tensorflow/supervised/eval*@
__saved_model_init_op'%
__saved_model_init_op
init_1Ů
Đ
:
Add
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
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
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
9
Softmax
logits"T
softmax"T"
Ttype:
2
:
Sub
x"T
y"T
z"T"
Ttype:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape
9
VarIsInitializedOp
resource
is_initialized
"serve*2.0.0-alpha02v1.12.0-9492-g2c319fb4158ţ
s
dense_12_inputPlaceholder*
shape:˙˙˙˙˙˙˙˙˙*
dtype0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
0dense_12/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@dense_12/kernel*
valueB"     *
dtype0*
_output_shapes
:

.dense_12/kernel/Initializer/random_uniform/minConst*"
_class
loc:@dense_12/kernel*
valueB
 *HY˝*
dtype0*
_output_shapes
: 

.dense_12/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@dense_12/kernel*
valueB
 *HY=*
dtype0*
_output_shapes
: 
×
8dense_12/kernel/Initializer/random_uniform/RandomUniformRandomUniform0dense_12/kernel/Initializer/random_uniform/shape*
T0*"
_class
loc:@dense_12/kernel*
dtype0* 
_output_shapes
:

Ú
.dense_12/kernel/Initializer/random_uniform/subSub.dense_12/kernel/Initializer/random_uniform/max.dense_12/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_12/kernel*
_output_shapes
: 
î
.dense_12/kernel/Initializer/random_uniform/mulMul8dense_12/kernel/Initializer/random_uniform/RandomUniform.dense_12/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@dense_12/kernel* 
_output_shapes
:

ŕ
*dense_12/kernel/Initializer/random_uniformAdd.dense_12/kernel/Initializer/random_uniform/mul.dense_12/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_12/kernel* 
_output_shapes
:

 
dense_12/kernelVarHandleOp*
shape:
* 
shared_namedense_12/kernel*"
_class
loc:@dense_12/kernel*
dtype0*
_output_shapes
: 
o
0dense_12/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_12/kernel*
_output_shapes
: 

dense_12/kernel/AssignAssignVariableOpdense_12/kernel*dense_12/kernel/Initializer/random_uniform*"
_class
loc:@dense_12/kernel*
dtype0

#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*"
_class
loc:@dense_12/kernel*
dtype0* 
_output_shapes
:


dense_12/bias/Initializer/zerosConst* 
_class
loc:@dense_12/bias*
valueB*    *
dtype0*
_output_shapes	
:

dense_12/biasVarHandleOp*
shape:*
shared_namedense_12/bias* 
_class
loc:@dense_12/bias*
dtype0*
_output_shapes
: 
k
.dense_12/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_12/bias*
_output_shapes
: 

dense_12/bias/AssignAssignVariableOpdense_12/biasdense_12/bias/Initializer/zeros* 
_class
loc:@dense_12/bias*
dtype0

!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias* 
_class
loc:@dense_12/bias*
dtype0*
_output_shapes	
:
p
dense_12/MatMul/ReadVariableOpReadVariableOpdense_12/kernel*
dtype0* 
_output_shapes
:

|
dense_12/MatMulMatMuldense_12_inputdense_12/MatMul/ReadVariableOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
j
dense_12/BiasAdd/ReadVariableOpReadVariableOpdense_12/bias*
dtype0*
_output_shapes	
:

dense_12/BiasAddBiasAdddense_12/MatMuldense_12/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Z
dense_12/ReluReludense_12/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
dropout_6/IdentityIdentitydense_12/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
0dense_13/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@dense_13/kernel*
valueB"   
   *
dtype0*
_output_shapes
:

.dense_13/kernel/Initializer/random_uniform/minConst*"
_class
loc:@dense_13/kernel*
valueB
 *Ű˝*
dtype0*
_output_shapes
: 

.dense_13/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@dense_13/kernel*
valueB
 *Ű=*
dtype0*
_output_shapes
: 
Ö
8dense_13/kernel/Initializer/random_uniform/RandomUniformRandomUniform0dense_13/kernel/Initializer/random_uniform/shape*
T0*"
_class
loc:@dense_13/kernel*
dtype0*
_output_shapes
:	

Ú
.dense_13/kernel/Initializer/random_uniform/subSub.dense_13/kernel/Initializer/random_uniform/max.dense_13/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_13/kernel*
_output_shapes
: 
í
.dense_13/kernel/Initializer/random_uniform/mulMul8dense_13/kernel/Initializer/random_uniform/RandomUniform.dense_13/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@dense_13/kernel*
_output_shapes
:	

ß
*dense_13/kernel/Initializer/random_uniformAdd.dense_13/kernel/Initializer/random_uniform/mul.dense_13/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_13/kernel*
_output_shapes
:	


dense_13/kernelVarHandleOp*
shape:	
* 
shared_namedense_13/kernel*"
_class
loc:@dense_13/kernel*
dtype0*
_output_shapes
: 
o
0dense_13/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_13/kernel*
_output_shapes
: 

dense_13/kernel/AssignAssignVariableOpdense_13/kernel*dense_13/kernel/Initializer/random_uniform*"
_class
loc:@dense_13/kernel*
dtype0

#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*"
_class
loc:@dense_13/kernel*
dtype0*
_output_shapes
:	


dense_13/bias/Initializer/zerosConst* 
_class
loc:@dense_13/bias*
valueB
*    *
dtype0*
_output_shapes
:


dense_13/biasVarHandleOp*
shape:
*
shared_namedense_13/bias* 
_class
loc:@dense_13/bias*
dtype0*
_output_shapes
: 
k
.dense_13/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_13/bias*
_output_shapes
: 

dense_13/bias/AssignAssignVariableOpdense_13/biasdense_13/bias/Initializer/zeros* 
_class
loc:@dense_13/bias*
dtype0

!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias* 
_class
loc:@dense_13/bias*
dtype0*
_output_shapes
:

o
dense_13/MatMul/ReadVariableOpReadVariableOpdense_13/kernel*
dtype0*
_output_shapes
:	


dense_13/MatMulMatMuldropout_6/Identitydense_13/MatMul/ReadVariableOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

i
dense_13/BiasAdd/ReadVariableOpReadVariableOpdense_13/bias*
dtype0*
_output_shapes
:


dense_13/BiasAddBiasAdddense_13/MatMuldense_13/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

_
dense_13/SoftmaxSoftmaxdense_13/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

-
predict/group_depsNoOp^dense_13/Softmax
U
ConstConst"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_1Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_2Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_3Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_4Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
\
Const_5Const"/device:CPU:0*
valueB Bmodel*
dtype0*
_output_shapes
: 
W
Const_6Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_7Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_8Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_9Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_10Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
¤
RestoreV2/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
r
RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

	RestoreV2	RestoreV2Const_5RestoreV2/tensor_namesRestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
B
IdentityIdentity	RestoreV2*
T0*
_output_shapes
:
L
AssignVariableOpAssignVariableOpdense_12/kernelIdentity*
dtype0
¤
RestoreV2_1/tensor_namesConst"/device:CPU:0*I
value@B>B4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_1	RestoreV2Const_5RestoreV2_1/tensor_namesRestoreV2_1/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
F

Identity_1IdentityRestoreV2_1*
T0*
_output_shapes
:
N
AssignVariableOp_1AssignVariableOpdense_12/bias
Identity_1*
dtype0
Ś
RestoreV2_2/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_2/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_2	RestoreV2Const_5RestoreV2_2/tensor_namesRestoreV2_2/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
F

Identity_2IdentityRestoreV2_2*
T0*
_output_shapes
:
P
AssignVariableOp_2AssignVariableOpdense_13/kernel
Identity_2*
dtype0
¤
RestoreV2_3/tensor_namesConst"/device:CPU:0*I
value@B>B4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_3/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_3	RestoreV2Const_5RestoreV2_3/tensor_namesRestoreV2_3/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
F

Identity_3IdentityRestoreV2_3*
T0*
_output_shapes
:
N
AssignVariableOp_3AssignVariableOpdense_13/bias
Identity_3*
dtype0
O
VarIsInitializedOpVarIsInitializedOpdense_13/bias*
_output_shapes
: 
S
VarIsInitializedOp_1VarIsInitializedOpdense_13/kernel*
_output_shapes
: 
Q
VarIsInitializedOp_2VarIsInitializedOpdense_12/bias*
_output_shapes
: 
S
VarIsInitializedOp_3VarIsInitializedOpdense_12/kernel*
_output_shapes
: 
l
initNoOp^dense_12/bias/Assign^dense_12/kernel/Assign^dense_13/bias/Assign^dense_13/kernel/Assign
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
dtype0*
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
dtype0*
_output_shapes
: 
¨	
save/Const_1Const*ë
valueáBŢ B×{"class_name": "Sequential", "config": {"layers": [{"class_name": "Dense", "config": {"activation": "relu", "activity_regularizer": null, "batch_input_shape": [null, 784], "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "dtype": "float32", "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "kernel_regularizer": null, "name": "dense_12", "trainable": true, "units": 512, "use_bias": true}}, {"class_name": "Dropout", "config": {"dtype": "float32", "name": "dropout_6", "noise_shape": null, "rate": 0.2, "seed": null, "trainable": true}}, {"class_name": "Dense", "config": {"activation": "softmax", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "dtype": "float32", "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "kernel_regularizer": null, "name": "dense_13", "trainable": true, "units": 10, "use_bias": true}}], "name": "sequential_6"}}*
dtype0*
_output_shapes
: 
Ú
save/Const_2Const*
valueB B{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 784], "dtype": "float32", "name": "dense_12_input", "sparse": false}}*
dtype0*
_output_shapes
: 
â
save/Const_3Const*Ľ
valueB B{"class_name": "Dropout", "config": {"dtype": "float32", "name": "dropout_6", "noise_shape": null, "rate": 0.2, "seed": null, "trainable": true}}*
dtype0*
_output_shapes
: 

save/Const_4Const*Ţ
valueÔBŃ BĘ{"class_name": "Dense", "config": {"activation": "relu", "activity_regularizer": null, "batch_input_shape": [null, 784], "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "dtype": "float32", "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "kernel_regularizer": null, "name": "dense_12", "trainable": true, "units": 512, "use_bias": true}}*
dtype0*
_output_shapes
: 
ű
save/Const_5Const*ž
value´Bą BŞ{"class_name": "Dense", "config": {"activation": "softmax", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "dtype": "float32", "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "kernel_regularizer": null, "name": "dense_13", "trainable": true, "units": 10, "use_bias": true}}*
dtype0*
_output_shapes
: 

save/SaveV2/tensor_namesConst*Ě
valueÂBż	B/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-2/.ATTRIBUTES/OBJECT_CONFIG_JSONB3layer_with_weights-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-1/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:	
u
save/SaveV2/shape_and_slicesConst*%
valueB	B B B B B B B B B *
dtype0*
_output_shapes
:	
Č
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicessave/Const_1save/Const_2save/Const_3save/Const_4!dense_12/bias/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOpsave/Const_5!dense_13/bias/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp*
dtypes
2	
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
Ť
save/RestoreV2/tensor_namesConst"/device:CPU:0*Ě
valueÂBż	B/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-2/.ATTRIBUTES/OBJECT_CONFIG_JSONB3layer_with_weights-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-1/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:	

save/RestoreV2/shape_and_slicesConst"/device:CPU:0*%
valueB	B B B B B B B B B *
dtype0*
_output_shapes
:	
Ç
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2	*8
_output_shapes&
$:::::::::

	save/NoOpNoOp

save/NoOp_1NoOp

save/NoOp_2NoOp

save/NoOp_3NoOp
N
save/IdentityIdentitysave/RestoreV2:4*
T0*
_output_shapes
:
T
save/AssignVariableOpAssignVariableOpdense_12/biassave/Identity*
dtype0
P
save/Identity_1Identitysave/RestoreV2:5*
T0*
_output_shapes
:
Z
save/AssignVariableOp_1AssignVariableOpdense_12/kernelsave/Identity_1*
dtype0

save/NoOp_4NoOp
P
save/Identity_2Identitysave/RestoreV2:7*
T0*
_output_shapes
:
X
save/AssignVariableOp_2AssignVariableOpdense_13/biassave/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:8*
T0*
_output_shapes
:
Z
save/AssignVariableOp_3AssignVariableOpdense_13/kernelsave/Identity_3*
dtype0
Â
save/restore_allNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_2^save/AssignVariableOp_3
^save/NoOp^save/NoOp_1^save/NoOp_2^save/NoOp_3^save/NoOp_4

init_1NoOp"D
save/Const:0save/control_dependency:0save/restore_all 5 @F8"
trainable_variablesűř

dense_12/kernel:0dense_12/kernel/Assign%dense_12/kernel/Read/ReadVariableOp:0(2,dense_12/kernel/Initializer/random_uniform:08
s
dense_12/bias:0dense_12/bias/Assign#dense_12/bias/Read/ReadVariableOp:0(2!dense_12/bias/Initializer/zeros:08

dense_13/kernel:0dense_13/kernel/Assign%dense_13/kernel/Read/ReadVariableOp:0(2,dense_13/kernel/Initializer/random_uniform:08
s
dense_13/bias:0dense_13/bias/Assign#dense_13/bias/Read/ReadVariableOp:0(2!dense_13/bias/Initializer/zeros:08"
	variablesűř

dense_12/kernel:0dense_12/kernel/Assign%dense_12/kernel/Read/ReadVariableOp:0(2,dense_12/kernel/Initializer/random_uniform:08
s
dense_12/bias:0dense_12/bias/Assign#dense_12/bias/Read/ReadVariableOp:0(2!dense_12/bias/Initializer/zeros:08

dense_13/kernel:0dense_13/kernel/Assign%dense_13/kernel/Read/ReadVariableOp:0(2,dense_13/kernel/Initializer/random_uniform:08
s
dense_13/bias:0dense_13/bias/Assign#dense_13/bias/Read/ReadVariableOp:0(2!dense_13/bias/Initializer/zeros:08*Ł
serving_default
:
dense_12_input(
dense_12_input:0˙˙˙˙˙˙˙˙˙5
dense_13)
dense_13/Softmax:0˙˙˙˙˙˙˙˙˙
tensorflow/serving/predict*@
__saved_model_init_op'%
__saved_model_init_op
init_1
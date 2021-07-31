#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   icp_ED_LM.py
@Contact :   chengc0611@gmail.com
@License :   

@Modify Time      @Author       @Version    @Desciption
------------      -------       --------    -----------
7/29/2021 3:13 PM   Cheng Chen    1.0         None
'''

# import lib
import itk


def reg_object_type_def(dimension=3):
    # Define point types
    PointType = itk.Point[itk.F, dimension]
    PointSetType = itk.PointSet[itk.F, dimension]

    # define transform type
    TransformType = itk.Euler3DTransform  # [itk.D]

    # define metric type
    PointSetMetricType = itk.JensenHavrdaCharvatTsallisPointSetToPointSetMetricv4[PointSetType]

    ShiftScalesType = itk.RegistrationParameterScalesFromPhysicalShift[PointSetMetricType]

    OptimizerType = itk.LevenbergMarquardtOptimizer

    return TransformType, PointSetMetricType, ShiftScalesType, OptimizerType


def reg_object_init(TransformType,
                    PointSetMetricType, fixed_set, moving_set,
                    ShiftScalesType,
                    OptimizerType, num_iterations):

    transform = TransformType.New()
    transform.SetIdentity()

    metric = PointSetMetricType.New(
        FixedPointSet=fixed_set,
        MovingPointSet=moving_set,
        PointSetSigma=1.0,
        KernelSigma=10.0,
        UseAnisotropicCovariances=True,
        CovarianceKNeighborhood=5,
        EvaluationKNeighborhood=50,
        MovingTransform=transform,
        Alpha=1.0)
    metric.Initialize()

    shift_scale_estimator = ShiftScalesType.New(
        Metric=metric,
        VirtualDomainPointSet=metric.GetVirtualTransformedPointSet())

    optimizer = OptimizerType.New(
        Metric=metric,
        NumberOfIterations=num_iterations,
        ScalesEstimator=shift_scale_estimator,
    )

    return metric, shift_scale_estimator, optimizer

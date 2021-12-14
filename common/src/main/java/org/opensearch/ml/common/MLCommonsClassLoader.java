/*
 *  SPDX-License-Identifier: Apache-2.0
 *
 *  The OpenSearch Contributors require contributions made to
 *  this file be licensed under the Apache-2.0 license or a
 *  compatible open source license.
 *
 *  Modifications Copyright OpenSearch Contributors. See
 *  GitHub history for details.
 */

package org.opensearch.ml.common;

import org.apache.commons.beanutils.BeanUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.ml.common.annotation.InputDataSet;
import org.opensearch.ml.common.annotation.Output;
import org.opensearch.ml.common.annotation.Parameter;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.input.dataset.MLInputDataType;
import org.opensearch.ml.common.output.MLOutput;
import org.opensearch.ml.common.output.OutputType;
import org.reflections.Reflections;

import java.lang.reflect.Constructor;
import java.security.AccessController;
import java.security.PrivilegedActionException;
import java.security.PrivilegedExceptionAction;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;


public class MLCommonsClassLoader {

    private static final Logger logger = LogManager.getLogger(MLCommonsClassLoader.class);
    private static Map<Enum<?>, Class<?>> parameterClassMap = new HashMap<>();
    private static Map<Enum<?>, Class<?>> inputDataSetClassMap = new HashMap<>();
    private static Map<Enum<?>, Class<?>> inputClassMap = new HashMap<>();
    private static Map<Enum<?>, Class<?>> outputClassMap = new HashMap<>();

    static {
        try {
            AccessController.doPrivileged((PrivilegedExceptionAction<Void>) () -> {
                loadClassMapping();
                return null;
            });
        } catch (PrivilegedActionException e) {
            throw new RuntimeException("Can't load class mapping in ML commons", e);
        }
    }

    public static void loadClassMapping() {
        loadMLAlgoParameterClassMapping();
        loadMLInputDataSetClassMapping();
        loadOutputClassMapping();
    }

    /**
     * Load ML algorithm parameter and ML output class.
     */
    private static void loadMLAlgoParameterClassMapping() {
        Reflections reflections = new Reflections("org.opensearch.ml.common.parameter");

        Set<Class<?>> classes = reflections.getTypesAnnotatedWith(Parameter.class);
        // Load ML algorithm parameter class
        for (Class<?> clazz : classes) {
            Parameter mlAlgoParameter = clazz.getAnnotation(Parameter.class);
            FunctionName[] algorithms = mlAlgoParameter.algorithms();
            if (algorithms != null && algorithms.length > 0) {
                for(FunctionName name : algorithms){
                    parameterClassMap.put(name, clazz);
                }
            }
        }
    }


    /**
     * Load ML input data set class
     */
    private static void loadMLInputDataSetClassMapping() {
        Reflections reflections = new Reflections("org.opensearch.ml.common.input.dataset");
        Set<Class<?>> classes = reflections.getTypesAnnotatedWith(InputDataSet.class);
        for (Class<?> clazz : classes) {
            InputDataSet inputDataSet = clazz.getAnnotation(InputDataSet.class);
            MLInputDataType value = inputDataSet.value();
            if (value != null) {
                inputDataSetClassMap.put(value, clazz);
            }
        }
    }

    private static void loadOutputClassMapping() {
        Reflections reflections = new Reflections("org.opensearch.ml.common.output");

        // Load ML output class
        Set<Class<?>> classes = reflections.getTypesAnnotatedWith(Output.class);
        for (Class<?> clazz : classes) {
            Output mlAlgoOutput = clazz.getAnnotation(Output.class);
            OutputType mlOutputType = mlAlgoOutput.value();
            if (mlOutputType != null) {
                outputClassMap.put(mlOutputType, clazz);
            }
        }
    }


    @SuppressWarnings("unchecked")
    public static <T extends Enum<T>, S, I extends Object> S initInstance(T type, I in, Class<?> constructorParamClass) {
        Class<?> clazz = parameterClassMap.get(type);
        if (clazz == null) {
            throw new IllegalArgumentException("Can't find class for type " + type);
        }
        try {
            Constructor<?> constructor = clazz.getConstructor(constructorParamClass);
            return (S) constructor.newInstance(in);
        } catch (Exception e) {
            logger.error("Failed to init instance for type " + type, e);
            return null;
        }
    }

    public static <T extends Enum<T>, S> S initInstance(T type, List<Object> params) {
        return initInstance(type, params, (Map<String, Object>)null);
    }

    public static <T extends Enum<T>, S> S initInstance(T type, List<Object> params, Map<String, Object> properties) {
        Class<?> clazz = parameterClassMap.get(type);
        return initInstance(type, params, properties, clazz);
    }

    public static <T extends Enum<T>, S> S initFunctionInput(T type, List<Object> params, Map<String, Object> properties) {
        Class<?> clazz = inputClassMap.get(type);
        if (clazz == null) {
            clazz = MLInput.class;
        }
        return initInstance(type, params, properties, clazz);
    }

    public static <T extends Enum<T>, S> S initFunctionOutput(T type, List<Object> params, Map<String, Object> properties) {
        Class<?> clazz = inputClassMap.get(type);
        if (clazz == null) {
            clazz = MLOutput.class;
        }
        return initInstance(type, params, properties, clazz);
    }

    private static <T extends Enum<T>, S> S initInstance(T type, List<Object> params, Map<String, Object> properties, Class<?> clazz) {
        if (clazz == null) {
            throw new IllegalArgumentException("Can't find class for type " + type);
        }
        try {
            Constructor<?> constructor;
            S instance;
            if (params == null || params.size() == 0) {
                constructor = clazz.getConstructor();
                instance = (S) constructor.newInstance();
            } else {
                List<? extends Class<?>> constructorParams = params.stream().map(t -> t.getClass()).collect(Collectors.toList());
                constructor = clazz.getConstructor(constructorParams.toArray(new Class<?>[0]));
                instance = (S) constructor.newInstance(params.toArray(new Object[0]));
            }
            BeanUtils.populate(instance, properties);
            return instance;
        } catch (Exception e) {
            logger.error("Failed to init instance for type " + type, e);
            return null;
        }
    }

}

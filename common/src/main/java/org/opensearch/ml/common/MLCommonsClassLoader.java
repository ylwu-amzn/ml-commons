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
import org.opensearch.ml.common.annotation.FunctionInput;
import org.opensearch.ml.common.annotation.InputDataSet;
import org.opensearch.ml.common.annotation.MLAlgoOutput;
import org.opensearch.ml.common.annotation.MLAlgoParameter;
import org.opensearch.ml.common.input.dataset.MLInputDataType;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.output.MLOutput;
import org.opensearch.ml.common.output.MLOutputType;
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
    private static Map<Enum<?>, Class<?>> functionInputClassMap = new HashMap<>();
    private static Map<Enum<?>, Class<?>> functionOutputClassMap = new HashMap<>();

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
    }

    /**
     * Load ML algorithm parameter and ML output class.
     */
    private static void loadMLAlgoParameterClassMapping() {
        Reflections reflections = new Reflections("org.opensearch.ml.common.parameter");

        Set<Class<?>> classes = reflections.getTypesAnnotatedWith(MLAlgoParameter.class);
        // Load ML algorithm parameter class
        for (Class<?> clazz : classes) {
            MLAlgoParameter mlAlgoParameter = clazz.getAnnotation(MLAlgoParameter.class);
            FunctionName[] algorithms = mlAlgoParameter.algorithms();
            if (algorithms != null && algorithms.length > 0) {
                for(FunctionName name : algorithms){
                    parameterClassMap.put(name, clazz);
                }
            }
        }

        // Load Function input class
        classes = reflections.getTypesAnnotatedWith(FunctionInput.class);
        for (Class<?> clazz : classes) {
            FunctionInput functionInput = clazz.getAnnotation(FunctionInput.class);
            FunctionName[] functions = functionInput.functions();
            if (functions != null && functions.length > 0) {
                for(FunctionName name : functions){
                    functionInputClassMap.put(name, clazz);
                }
            }
        }

        // Load ML output class
        classes = reflections.getTypesAnnotatedWith(MLAlgoOutput.class);
        for (Class<?> clazz : classes) {
            MLAlgoOutput mlAlgoOutput = clazz.getAnnotation(MLAlgoOutput.class);
            MLOutputType mlOutputType = mlAlgoOutput.value();
            if (mlOutputType != null) {
                parameterClassMap.put(mlOutputType, clazz);
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
                parameterClassMap.put(value, clazz);
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

//    @SuppressWarnings("unchecked")
//    public static <T extends Enum<T>, S> S initInstance(T type, List<Tuple<Class<?>, Object>> params, Map<String, Object> properties) {
//        Class<?> clazz = parameterClassMap.get(type);
//        if (clazz == null) {
//            throw new IllegalArgumentException("Can't find class for type " + type);
//        }
//        try {
//            Constructor<?> constructor;
//            S instance;
//            if (params == null || params.size() == 0) {
//                constructor = clazz.getConstructor();
//                instance = (S) constructor.newInstance();
//            } else {
//                List<? extends Class<?>> constructorParams = params.stream().map(t -> t.v1()).collect(Collectors.toList());
//                constructor = clazz.getConstructor(constructorParams.toArray(new Class<?>[0]));
//                List<Object> paramValues = params.stream().map(t -> t.v2()).collect(Collectors.toList());
//                instance = (S) constructor.newInstance(paramValues.toArray(new Object[0]));
//            }
//            BeanUtils.populate(instance, properties);
//            return instance;
//        } catch (Exception e) {
//            logger.error("Failed to init instance for type " + type, e);
//            return null;
//        }
//    }

    public static <T extends Enum<T>, S> S initInstance(T type, List<Object> params) {
        return initInstance(type, params, (Map<String, Object>)null);
    }

    public static <T extends Enum<T>, S> S initInstance(T type, List<Object> params, Map<String, Object> properties) {
        Class<?> clazz = parameterClassMap.get(type);
        return initInstance(type, params, properties, clazz);
    }

    public static <T extends Enum<T>, S> S initFunctionInput(T type, List<Object> params, Map<String, Object> properties) {
        Class<?> clazz = functionInputClassMap.get(type);
        if (clazz == null) {
            clazz = MLInput.class;
        }
        return initInstance(type, params, properties, clazz);
    }

    public static <T extends Enum<T>, S> S initFunctionOutput(T type, List<Object> params, Map<String, Object> properties) {
        Class<?> clazz = functionOutputClassMap.get(type);
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

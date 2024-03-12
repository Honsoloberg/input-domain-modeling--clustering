package com.butler.clustering;

import java.io.File;

import net.sf.javaml.clustering.evaluation.AICScore;
import net.sf.javaml.clustering.evaluation.BICScore;
import net.sf.javaml.clustering.evaluation.ClusterEvaluation;
import net.sf.javaml.clustering.evaluation.SumOfAveragePairwiseSimilarities;
import net.sf.javaml.clustering.evaluation.SumOfSquaredErrors;

import net.sf.javaml.clustering.AQBC;
import net.sf.javaml.clustering.Clusterer;
import net.sf.javaml.clustering.KMeans;
import net.sf.javaml.clustering.DensityBasedSpatialClustering;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.tools.data.FileHandler;

public class App 
{
    public static void main(String[] args) throws Exception
    {
    	//iris dataset
        Dataset iris = FileHandler.loadDataset(new File("iris.data"), 4, ",");
        
        //declaration of clustering machine learning methods (i=_ for all arrays)
        Clusterer kmean = new KMeans(); //i=0
        Clusterer adaptive = new AQBC(); //i=1
        Clusterer spatial = new DensityBasedSpatialClustering(); //i=2
        
        //datasets for each ML method
        Dataset[] method1 = null;
        Dataset[] method2 = null;
        Dataset[] method3 = null;
        
        //total time for each method
        long[] totalTime = new long[3];
        
        long start = 0;
        long end = 0;
        
        //Cluster Execution ----------------------
        start = System.nanoTime();
        method1 = kmean.cluster(iris);
        end = System.nanoTime();
        totalTime[0] = end - start;
        
        start = System.nanoTime();
        method2 = adaptive.cluster(iris);
        end = System.nanoTime();
        totalTime[1] = end - start;
        
        start = System.nanoTime();
        method3 = spatial.cluster(iris);
        end = System.nanoTime();
        totalTime[2] = end - start;
        
        //Cluster Evaluation ----------------------
        ClusterEvaluation aic = new AICScore();
        ClusterEvaluation bic = new BICScore();
        ClusterEvaluation squared = new SumOfSquaredErrors();
        ClusterEvaluation pairwise = new SumOfAveragePairwiseSimilarities();
        
        double[] aicScore = new double[3];
        double[] bicScore = new double[3];
        double[] sseScore = new double[3];
        double[] sapsScore = new double[3];
        
        aicScore[0] = aic.score(method1);
        bicScore[0] = bic.score(method1);
        sseScore[0] = squared.score(method1);
        sapsScore[0] = pairwise.score(method1);
        
        aicScore[1] = aic.score(method2);
        bicScore[1] = bic.score(method2);
        sseScore[1] = squared.score(method2);
        sapsScore[1] = pairwise.score(method2);
        
        aicScore[2] = aic.score(method3);
        bicScore[2] = bic.score(method3);
        sseScore[2] = squared.score(method3);
        sapsScore[2] = pairwise.score(method3);
        
        System.out.println("\n--------------------------------------------------------------------------------------------");
        for(int i=0; i < method1.length; i++) {
        	System.out.println(method1[i]);
        }
        System.out.println("\n");
        for(int i=0; i < method2.length; i++) {
        	System.out.println(method2[i]);
        }
        System.out.println("\n");
        for(int i=0; i < method3.length; i++) {
        	System.out.println(method3[i]);
        }
        
        System.out.println("\n--------------------------------------------------------------------------------------------\n");
        System.out.println("\t\t\tKMean\t\tAQBC\t\tDBSS");
        System.out.printf("Cluster Count: \t\t" + "%d" + "\t\t" + "%d" + "\t\t" + "%d\n", method1.length, method1.length, method1.length);
        System.out.printf("AIC Score: \t\t" + "%.2f" + "\t\t" + "%.2f" + "\t\t" + "%.2f\n", aicScore[0], aicScore[1], aicScore[2]);
        System.out.printf("BIC Score: \t\t" + "%.2f" + "\t\t" + "%.2f" + "\t\t" + "%.2f\n", bicScore[0], bicScore[1], bicScore[2]);
        System.out.printf("Sum of Squared Errors: \t" + "%.2f" + "\t\t" + "%.2f" + "\t\t" + "%.2f\n", sseScore[0], sseScore[1], sseScore[2]);
        System.out.printf("Pairwise Similarities: \t" + "%.2f" + "\t\t" + "%.2f" + "\t\t" + "%.2f\n", sapsScore[0], sapsScore[1], sapsScore[2]);
        System.out.printf("Excution Time(ms): \t" + "%d" + "\t\t" + "%d" + "\t\t" + "%d\n", totalTime[0]/100000, totalTime[1]/100000, totalTime[2]/100000);
        System.out.println("\n--------------------------------------------------------------------------------------------\n");
    }
}

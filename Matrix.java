package com.opencv;

import org.opencv.core.Core;
import org.opencv.core.*;
import org.opencv.dnn.*;
import org.opencv.dnn.Net;
import org.opencv.utils.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;

public class Matrix {

        private static List<String> getOutputNames(Net net) {


            List<String> names = new ArrayList<>();
            List<Integer> outLayers = net.getUnconnectedOutLayers().toList();
            List<String> layersNames = net.getLayerNames();

            outLayers.forEach((item) -> names.add(layersNames.get(item - 1)));
            return names;
        }

        public static void main(String[] args)  throws Exception {
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

            String modelWeights = "C:/yolo/yolov3.weights";
            String modelConfiguration = "C:/yolo/yolov3.cfg";
            String modelNames = "C:/yolo/coco.names";


            ArrayList<String> classes = new ArrayList<>();
            FileReader file = new FileReader(modelNames);
            BufferedReader bufferedReader = new BufferedReader(file);
            String Line;
            while ((Line = bufferedReader.readLine()) != null) {
                classes.add(Line);
            }
            bufferedReader.close();




            Net net = Dnn.readNetFromDarknet(modelConfiguration, modelWeights);

            //object localisation components
            Mat image = Imgcodecs.imread("room.jpg");
            Size sz = new Size(416, 416);
            Mat blob = Dnn.blobFromImage(image, 0.00392, sz, new Scalar(0), true, false);
            net.setInput(blob);

            List<Mat> result = new ArrayList<>();
            List<String> outBlobNames = getOutputNames(net);

            net.forward(result, outBlobNames);


           // outBlobNames.forEach(System.out::println);
            //result.forEach(System.out::println);
            System.out.println(result);

            float confThreshold = 0.55f;

            LinkedList<Integer> clsIds = new LinkedList<>();
            List<Float> confs = new ArrayList<>();
            List<Rect> rects = new ArrayList<>();


            //object classification components

            for (int i = 0; i < result.size(); i++)
            {

                // each row is a candidate detection, the 1st 4 numbers are
                // [center_x, center_y, width, height], followed by (N-4) class probabilities
                Mat level = result.get(i);
                for (int j = 0; j < level.rows(); ++j)
                {
                    Mat row = level.row(j);
                    Mat scores = row.colRange(5, level.cols());

                    Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
                    float confidence = (float)mm.maxVal;
                    Point classIdPoint = mm.maxLoc;

                    if (confidence > confThreshold)
                    {
                        int centerX = (int)(row.get(0,0)[0] * image.cols());
                        int centerY = (int)(row.get(0,1)[0] * image.rows());
                        int width   = (int)(row.get(0,2)[0] * image.cols());
                        int height  = (int)(row.get(0,3)[0] * image.rows());
                        int left    = centerX - width  / 2;
                        int top     = centerY - height / 2;


                        clsIds.addLast( (int)classIdPoint.x );
                        confs.add(confidence);
                        rects.add(new Rect(left, top, width, height));

                    }
                }
            }
            System.out.println(clsIds.toString()+"\n");
            System.out.println(classes.toString()+"\n");
            System.out.println(confs.toString()+"\n");
            System.out.println(rects.toString()+"\n");




            // Apply non-maximum suppression procedure.
            float nmsThresh = 0.55f;
            MatOfFloat confidences = new MatOfFloat(Converters.vector_float_to_Mat(confs));

            Rect[] boxesArray = rects.toArray(new Rect[0]);

            MatOfRect boxes = new MatOfRect(boxesArray);
            MatOfInt indices = new MatOfInt();

            Dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThresh, indices);



            // Draw result boxes:
            int [] ind = indices.toArray();

            for (int i = 0; i < ind.length; ++i)
            {
                int idx = ind[i];
                Rect box = boxesArray[idx];
                Imgproc.rectangle(image, box.tl(), box.br(), new Scalar(0,255,0), 2,20);

                String label = classes.get(clsIds.get(i)).toString();
                Imgproc.putText(image,label,new Point(box.x,box.y+30), 5, 5, Scalar.all(170),3);

               // System.out.println(box);
            }
            Imgcodecs.imwrite("out.png", image);

        }
    }



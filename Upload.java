package dark;

import java.awt.BorderLayout;
import java.awt.Image;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.filechooser.FileNameExtensionFilter;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;



public class Upload extends JFrame implements ActionListener{
	private JButton upload, detect;
	private JFileChooser choice;
	private JPanel container;
	private FileNameExtensionFilter imgFilter;
	private JTextField imgpth;
	private String imagepath;
	private String outputPath;
	private JLabel viewer;

	
	public Upload() {
		detect = new JButton("Detect and Classify");
		detect.setVisible(false);
		detect.addActionListener(this);
		viewer = new JLabel();
		imgpth = new JTextField();
		upload = new JButton("Upload image");
		upload.addActionListener(this);
		JPanel pane = new JPanel();
		JPanel pane2 = new JPanel();
		pane.add(upload);
		pane2.add(detect);
		container = new JPanel();
		choice = new JFileChooser();
		choice.setAcceptAllFileFilterUsed(false);
		imgFilter = new FileNameExtensionFilter("JPG & JPEG Images","JPG", "JPEG", "jpg", "jpeg");
		choice.setFileFilter(imgFilter);
		//choice.setFileSelectionMode(JFileChooser.FILES_ONLY);
		choice.setVisible(false);
		container.setLayout(new BorderLayout());
		container.add(choice);
		container.add(pane, BorderLayout.NORTH);
		container.add(viewer);
		container.add(pane2, BorderLayout.SOUTH);
		viewer.setVisible(false);
		
		this.setContentPane(container);
		this.setSize(500, 500);
		this.setTitle("Object detection in images");
		this.setLocationRelativeTo(null);
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		this.setVisible(true);
		
	}
	
	public static final String ALPHA_NUMERIC_STRING = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
	
	public static String randomAlphaNumeric(int count) {
	StringBuilder builder = new StringBuilder();
	while (count-- != 0) {
	int character = (int)(Math.random()*ALPHA_NUMERIC_STRING.length());
	builder.append(ALPHA_NUMERIC_STRING.charAt(character));
	}
	String str = builder.toString();
	return str;
	}
	
	public static List<String> getOutputNames(Net net) {
        List<String> names = new ArrayList<>();
        List<Integer> outLayers = net.getUnconnectedOutLayers().toList();
        List<String> layersNames = net.getLayerNames();

        outLayers.forEach((item) -> names.add(layersNames.get(item - 1)));
        return names;
	}
	public static void main(String[] args) {
		Upload up = new Upload();
	}
	@Override
	public void actionPerformed(ActionEvent e) {
		File file = null;
		Image img = null;
		if(e.getSource() == upload) {
			choice.setVisible(true);
			int outPut = choice.showOpenDialog(Upload.this);
			if (outPut == JFileChooser.APPROVE_OPTION) {
				try {
					file = choice.getSelectedFile();
                    imagepath = file.getAbsolutePath();
                    //imgpth.setText(imagepath);
                    //System.out.println(imagepath);
                    img = ImageIO.read(file);
                    //ImageIcon image = new ImageIcon(imagepath);
                    viewer.setIcon(new ImageIcon(img.getScaledInstance(499, 400, Image.SCALE_SMOOTH)));
                    //viewer.setIcon(image);
                    viewer.setVisible(true);
                    detect.setVisible(true);
                    
				}
				catch(IOException ex) {
					
				}
			}
		}
		if(e.getSource() == detect) {
			System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
			
			String modelWeights = "C:\\Users\\PC\\Desktop\\yolov3.weights";
	        String modelConfiguration = "C:\\Users\\PC\\Desktop\\yolov3.cfg";
	        
	        Net net = Dnn.readNetFromDarknet(modelConfiguration, modelWeights);
	        Mat image = Imgcodecs.imread(imagepath);
	        Size sz = new Size(416, 416);
	        Mat blob = Dnn.blobFromImage(image, 0.00392, sz, new Scalar(0), true, false);
	        net.setInput(blob);
	        List<Mat> result = new ArrayList<>();
	        List<String> outBlobNames = getOutputNames(net);

	        net.forward(result, outBlobNames);

	        outBlobNames.forEach(System.out::println);
	        result.forEach(System.out::println);

	        float confThreshold = 0.6f;
	        List<Integer> clsIds = new ArrayList<>();
	        List<Float> confs = new ArrayList<>();
	        List<Rect> rects = new ArrayList<>();
	        
	        for (int i = 0; i < result.size(); ++i)
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

	                    clsIds.add((int)classIdPoint.x);
	                    confs.add((float)confidence);
	                    rects.add(new Rect(left, top, width, height));
	                }
	            }
	        }
	        float nmsThresh = 0.5f;
	        MatOfFloat confidences = new MatOfFloat(Converters.vector_float_to_Mat(confs));
	        Rect[] boxesArray = rects.toArray(new Rect[0]);
	        MatOfRect boxes = new MatOfRect(boxesArray);
	        MatOfInt indices = new MatOfInt();
	        Dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThresh, indices);
	        int [] ind = indices.toArray();
	        for (int i = 0; i < ind.length; ++i)
	        {
	            int idx = ind[i];
	            Rect box = boxesArray[idx];
	            Imgproc.rectangle(image, box.tl(), box.br(), new Scalar(0,0,255), 2);
	            System.out.println(box);
	        }
	        outputPath = "C:\\Users\\PC\\Desktop\\"+ randomAlphaNumeric(6)+".png";
	        Imgcodecs.imwrite(outputPath, image);

		
			try {
				
				file = new File(outputPath);
				while(!(file.exists())) {
				
				}
				
				img = ImageIO.read(file);
				viewer.setIcon(new ImageIcon(img.getScaledInstance(499, 400, Image.SCALE_SMOOTH)));
				viewer.setVisible(true);
				
				
			}
			
		catch (IOException e1) {
				// TODO Auto-generated catch block
				System.out.println("Precessing....");
			}
		
			
		}
		
	}
}

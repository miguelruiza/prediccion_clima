
import weka.classifiers.Classifier;
import weka.core.SerializationHelper;
import weka.core.Instances;

import weka.core.Attribute;
import weka.core.DenseInstance;
import java.util.ArrayList;
import java.util.Scanner;

// Main class
// BreastCancer
public class Main {

    // Main driver method
    public static void main(String args[])
    {

        // Try block to check for exceptions
        try {


            // Cargar el modelo entrenado desde archivo (modificar la ruta en caso necesario)
            String archivo = "/home/miguelruiza/Documents/DOC ING SIS/DOC CS-409 Fundamentos/trabajo final/weka/RandomForest_split70_accuracy768_v4.model";
            Classifier classifier = (Classifier) SerializationHelper.read(archivo);


            // Definir la estructura de las instancias
            ArrayList<Attribute> attributes = new ArrayList<>();
            // Atributos numéricos
            attributes.add(new Attribute("mes"));
            attributes.add(new Attribute("maxC"));
            attributes.add(new Attribute("minC"));
            attributes.add(new Attribute("rocioC"));
            attributes.add(new Attribute("hum_rel"));
            attributes.add(new Attribute("velo_vien_kph"));
            attributes.add(new Attribute("vien_grad_sex"));
            attributes.add(new Attribute("nubosidad_por"));
            attributes.add(new Attribute("visibility"));
            attributes.add(new Attribute("radiac_solar_watios_m2"));
            attributes.add(new Attribute("energ_solar_MJ_m2"));
            attributes.add(new Attribute("uvindex"));
            // Atributo nominal
            ArrayList<String> classValues = new ArrayList<>();
            classValues.add("lluvioso");
            classValues.add("despejado");
            classValues.add("nublado");
            classValues.add("parcialmente_nublado");
            Attribute classAttribute = new Attribute("clase_hoy", classValues);
            attributes.add(classAttribute);
            // Crear el objeto Instances con la estructura definida
            Instances newData = new Instances("Clima Lima limpio analizar sin out smote randomize probar v4", attributes, 0);
            newData.setClass(classAttribute);
            // Establecer el índice de la clase
            newData.setClassIndex(newData.numAttributes() - 1);



            // Crear una nueva instancia utilizando DenseInstance
            DenseInstance instance1 = new DenseInstance(newData.numAttributes());
            instance1.setDataset(newData);



            // Solicitar interactivamente los valores de los atributos
            Scanner scanner = new Scanner(System.in);
            for (int i = 0; i < newData.numAttributes(); i++) {
                Attribute attribute = newData.attribute(i);

                if ( attribute.name() == "clase_hoy" ) {
                    System.out.print("Ingresa el valor para " + attribute.name() + " -> 0=lluvioso 1=despejado 2=nublado 3=parcialmente_nublado : ");
                } else {
                    System.out.print("Ingresa el valor para " + attribute.name() + ": ");
                }

                if (attribute.isNumeric()) {
                    double value = scanner.nextDouble();
                    instance1.setValue(attribute, value);
                } else {
                    int value = scanner.nextInt();
                    String value_ch = switch (value) {
                        case 0 -> "lluvioso";
                        case 1 -> "despejado";
                        case 2 -> "nublado";
                        case 3 -> "parcialmente_nublado";
                        default -> "";
                    };
                    System.out.println("Selecciono para " + attribute.name() + " -> " + value_ch);
                    instance1.setValue(attribute, value_ch);
                }
            }


            // Realizar la predicción para la primera instancia
            double predictedClass1 = classifier.classifyInstance(instance1);
            String predictedClassName1 = newData.classAttribute().value((int) predictedClass1);
            System.out.println("PREDICCIÓN CLIMA MAÑANA: " + predictedClassName1);


            // Cerrar el scanner
            scanner.close();

        }

        // Catch block to handle the exceptions
        catch (Exception e) {
            // Print message on the console
            System.out.println("Error Occurred!!!! \n"
                    + e.getMessage());
        }
    }
}
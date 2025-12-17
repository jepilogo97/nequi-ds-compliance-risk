# nequi-ds-compliance-risk

# 🏗️ 1. Arquitectura Analítica en AWS para Gestión Integral de Riesgos
## Análisis Dinámico de Riesgos Corporativos

Este repositorio documenta el **diseño de una arquitectura analítica conceptual en AWS** orientada a la **gestión integral de riesgos**, soportando análisis dinámico, modelos de Machine Learning, cálculos actuariales básicos y un fuerte enfoque en **trazabilidad, explicabilidad y gobierno del dato**.

---

## 🎯 Objetivos del Sistema

- Centralizar y gobernar información de riesgo proveniente de múltiples fuentes.
- Diseñar y mantener una **Matriz de Riesgo Corporativa** actualizable dinámicamente.
- Desarrollar **modelos de Machine Learning** para estimar la **probabilidad de eventos de riesgo**.
- Ejecutar **cálculos actuariales simples** (frecuencia, severidad, pérdida esperada, reservas y escenarios).
- Garantizar **trazabilidad end-to-end**, **explicabilidad de modelos** y **cumplimiento regulatorio**.


---

## 🧩 Diagrama de Arquitectura

![alt text](image.png)

![p1](https://github.com/user-attachments/assets/1879f646-76a3-488e-a644-a75184a1a690)

---

## 🗺️ Descripción de la Arquitectura por Fases

### 1️⃣ Fuentes de Datos
- Incidentes operativos  
- Sanciones regulatorias (PDF / XML)  
- PQRs (texto y audio WAV)  
- Exposición por producto  
- Eventos críticos (streaming)

Estas fuentes constituyen el origen de los eventos de riesgo y la exposición utilizada en la matriz de riesgo, los modelos predictivos y los cálculos actuariales.

---

### 2️⃣ Ingesta (Batch y Streaming)

**Batch**
- AWS DMS  
- AWS Transfer Family (SFTP)  
- AWS DataSync  
- API Gateway + WAF  

**Streaming**
- Kinesis Data Streams  
- Kinesis Firehose  
- EventBridge  

**Orquestación**
- AWS Step Functions

Esta capa garantiza una ingesta segura, desacoplada y totalmente trazable.

---

### 3️⃣ Data Lake (Bronze / Silver / Gold)

- **Bronze**: datos crudos e inmutables  
- **Silver**: datos limpios y estandarizados  
- **Gold**: datasets analíticos y de consumo  

Tecnologías:
- Amazon S3  
- Apache Iceberg (ACID, time-travel)  
- AWS Glue Data Catalog  

Aquí se almacenan la **matriz de riesgo corporativa**, resultados actuariales y features de ML.

---

### 4️⃣ Parsing y Enriquecimiento

- Textract (PDFs)  
- Transcribe (audio)  
- Comprehend (NLP)  
- AWS Lambda  

Convierte datos no estructurados en información analítica.

---

### 5️⃣ Calidad y Observabilidad

- Glue Data Quality / Deequ  
- Alertas SNS / Slack / Webhook  

Las métricas de calidad se almacenan en un **DQ Mart** consultable vía Athena o Redshift.

---

### 6️⃣ Cómputo y Analítica

- AWS Glue  
- Amazon EMR  
- Amazon Athena  
- Amazon Redshift  

Aquí se calculan:
- Matriz de riesgo corporativa  
- Cálculos actuariales  
- Data marts analíticos  

---

### 7️⃣ ML / MLOps

- Feature Store  
- SageMaker Pipelines  
- SageMaker Training  
- Model Registry  
- Batch Transform / Endpoints  
- Model Monitor  
- SageMaker Clarify  

Permite estimar probabilidades de eventos con explicabilidad y control de sesgo.

---

### 8️⃣ Consumo y Reporting

- Amazon QuickSight  
- Reportes regulados (PDF / CSV)  

Entrega información a comité de riesgos y auditoría.

---

### 9️⃣ Gobierno y Seguridad

- Lake Formation  
- KMS  
- Secrets Manager  
- CloudTrail  
- AWS Config  
- Security Hub  
- GuardDuty  
- Macie  
- CloudWatch  
- VPC / Endpoints  

Garantiza cumplimiento, seguridad y trazabilidad completa.

---

## 🔄 Flujo de Datos (Resumen)

1. Fuentes → Ingesta → S3 Bronze  
2. Parsing / Calidad → S3 Silver  
3. Analítica / ML → S3 Gold / Redshift  
4. Consumo → Dashboards y reportes  

---

## ✅ Consideraciones de Diseño

- Arquitectura event-driven
- Separación Bronze / Silver / Gold
- Time-travel y reproducibilidad con Iceberg
- Seguridad y gobierno alineados a entornos regulados

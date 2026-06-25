# 📊 Research & Search Findings (v1.01)

This document tracks the results of our candidate retrieval experiments and highlights the differences between our initial test implementations and our **True Architectural Intent** for the discovery pipeline.

---

## 🔍 Experiment v1.01: Simple Vector Binarization

In this initial run, we tested a baseline binarization approach on the candidate vectors.
*   **Model**: `all-MiniLM-L6-v2` (384 float32 dimensions)
*   **Method**: Basic thresholding to `int8` (0 or 1):
    $$\text{weight} > 0 \rightarrow 1 \quad \text{and} \quad \text{weight} \le 0 \rightarrow 0$$
*   **Metric**: Hamming distance (bitwise XOR) over the 100,000 candidate pool.

### 📈 Disk & Performance Metrics
*   💾 **Binarized Embedding File**: **36.8 MB** (extremely low footprint)
*   🗂️ **Candidate Index File**: **1.52 MB**
*   ⏱️ **Offline Embedding Creation**: **~1 hour**
*   🚀 **Stage 1 Recall Retrieval (Top 2000)**: **< 1 second**
    > Retrieval is **extremely fast** (sub-second). This speed suggests we can afford to use more computationally expensive matching or retrieval algorithms during Stage 1 if it improves recall accuracy.

---

### 📋 Top 5 Retrieved Profiles (v1.01)

Below is the CLI preview output for the top 5 candidates retrieved using this baseline method:

```bash
python src/test/preview_profiles.py CAND_0033138 CAND_0011245 CAND_0021628 CAND_0047338 CAND_0099907
```

#### 👤 [CAND_0033138] Aryan Reddy
*   **Headline**: Cloud Engineer | Full-stack development
*   **Location**: London, UK (7.2 Years Experience)
*   **Company**: Infosys (IT Services)

> **Summary**: Software engineer with 7.2 years of experience across web, backend, and cloud systems. Strong fundamentals in software development and system design. I've spent most of my career on web and API development — Python/Django and Node.js mostly. I've been keeping up with AI/ML at a self-learner level — taken some online courses, played with the OpenAI and Anthropic APIs, built a small RAG side project — but I haven't done it in a professional capacity yet. Open to roles where I can either deepen my software engineering work or, if the team is open to it, start contributing to ML-adjacent systems.

*   **Skills**: `[AWS (beginner)]` `[Node.js (intermediate)]` `[Angular (intermediate)]` `[Microservices (intermediate)]` `[MLflow (intermediate)]` `[OpenCV (intermediate)]` `[Azure (intermediate)]` `[Weaviate (intermediate)]` `[Rust (intermediate)]` `[ASR (intermediate)]` `[Next.js (intermediate)]` `[dbt (intermediate)]` `[Haystack (advanced)]` `[Data Pipelines (intermediate)]` `[Django (intermediate)]` `[Redis (intermediate)]`
*   **Experience**:
    *   **Cloud Engineer** at Infosys – 38 months *(Current)*
        *   Frontend engineering at a media company. React, TypeScript, and the typical surrounding tooling (Webpack, Jest, Cypress). Built the company's design system from scratch and led the migration from a legacy AngularJS app. Strong on the frontend craft — accessibility, performance, animations — but limited backend exposure.
    *   **QA Engineer** at Dunder Mifflin – 48 months
        *   Cloud infrastructure and DevOps work at an enterprise SaaS company. Owned the AWS account architecture (VPC, IAM, networking), the Terraform modules for our service deployments, and the Kubernetes cluster operations. Designed the CI/CD pipelines (GitLab CI + ArgoCD) and the monitoring stack (Prometheus, Grafana, Loki). Strong on the infra and ops side; haven't done much application development.
*   **Education**:
    *   **SRM Chennai** (tier_3) – Ph.D in Civil Engineering | Grade: 7.39 CGPA
    *   **Generic State University** (tier_4) – M.Tech in Machine Learning | Grade: 78%
*   **Signals**:
    *   **Expected Salary**: 12.5 - 17.4 LPA (INR) | **Notice Period**: 120 days
    *   **Work Mode**: remote | **Profile Completeness**: 49.7% | **GitHub Score**: 19.2

---

#### 👤 [CAND_0011245] Kiara Patel
*   **Headline**: Cloud Engineer | Full-stack development
*   **Location**: Trivandrum, Kerala, India (2.2 Years Experience)
*   **Company**: Dunder Mifflin (Paper Products)

> **Summary**: Software engineer with 2.2 years of experience across web, backend, and cloud systems. Strong fundamentals in software development and system design. I've spent most of my career on web and API development — Python/Django and Node.js mostly. I've been keeping up with AI/ML at a self-learner level — taken some online courses, played with the OpenAI and Anthropic APIs, built a small RAG side project — but I haven't done it in a professional capacity yet. Open to roles where I can either deepen my software engineering work or, if the team is open to it, start contributing to ML-adjacent systems.

*   **Skills**: `[Airflow (beginner)]` `[gRPC (intermediate)]` `[Snowflake (intermediate)]` `[Redis (beginner)]` `[Azure (beginner)]` `[Next.js (beginner)]` `[SEO (beginner)]` `[Recommendation Systems (advanced)]` `[AWS (beginner)]` `[Apache Beam (beginner)]` `[HTML (beginner)]` `[Microservices (intermediate)]`
*   **Experience**:
    *   **Cloud Engineer** at Dunder Mifflin – 26 months *(Current)*
        *   Frontend engineering at a media company. React, TypeScript, and the typical surrounding tooling (Webpack, Jest, Cypress). Built the company's design system from scratch and led the migration from a legacy AngularJS app. Strong on the frontend craft — accessibility, performance, animations — but limited backend exposure.
*   **Education**:
    *   **Christ University** (tier_3) – M.E. in Information Technology | Grade: 9.33 CGPA
*   **Signals**:
    *   **Expected Salary**: 20.1 - 33.7 LPA (INR) | **Notice Period**: 90 days
    *   **Work Mode**: hybrid | **Profile Completeness**: 50.9% | **GitHub Score**: -1

---

#### 👤 [CAND_0021628] Meera Goyal
*   **Headline**: Cloud Engineer | Full-stack development
*   **Location**: Indore, Madhya Pradesh, India (3.3 Years Experience)
*   **Company**: Cognizant (IT Services)

> **Summary**: Software engineer with 3.3 years of experience across web, backend, and cloud systems. Strong fundamentals in software development and system design. I've spent most of my career on web and API development — Python/Django and Node.js mostly. I've been keeping up with AI/ML at a self-learner level — taken some online courses, played with the OpenAI and Anthropic APIs, built a small RAG side project — but I haven't done it in a professional capacity yet. Open to roles where I can either deepen my software engineering work or, if the team is open to it, start contributing to ML-adjacent systems.

*   **Skills**: `[Redis (intermediate)]` `[Scrum (beginner)]` `[MongoDB (beginner)]` `[GraphQL (intermediate)]` `[Hadoop (intermediate)]` `[Semantic Search (intermediate)]` `[Go (intermediate)]` `[Project Management (beginner)]` `[Feature Engineering (advanced)]` `[Kubeflow (intermediate)]` `[gRPC (intermediate)]` `[Apache Flink (beginner)]` `[Figma (beginner)]` `[Time Series (advanced)]` `[Apache Beam (beginner)]` `[TensorFlow (advanced)]` `[Django (intermediate)]`
*   **Experience**:
    *   **Cloud Engineer** at Cognizant – 31 months *(Current)*
        *   Frontend engineering at a media company. React, TypeScript, and the typical surrounding tooling (Webpack, Jest, Cypress). Built the company's design system from scratch and led the migration from a legacy AngularJS app. Strong on the frontend craft — accessibility, performance, animations — but limited backend exposure.
    *   **QA Engineer** at Mphasis – 8 months
        *   Test automation and QA engineering for a fintech product. Built and maintained the end-to-end test suite using Selenium and pytest, plus the load-testing setup using Locust. Worked closely with developers on testability patterns and with product on acceptance criteria. Recent work has been on shifting test responsibility into the dev team — moving from QA-as-gate to QA-as-coach. Career has been entirely in QA/test engineering.
*   **Education**:
    *   **Generic State University** (tier_4) – M.E. in Physics | Grade: 8.65 CGPA
    *   **Symbiosis International** (tier_3) – M.Tech in Computer Engineering | Grade: 6.97 CGPA
*   **Signals**:
    *   **Expected Salary**: 16.4 - 19.3 LPA (INR) | **Notice Period**: 90 days
    *   **Work Mode**: onsite | **Profile Completeness**: 64.2% | **GitHub Score**: 17.2

---

#### 👤 [CAND_0047338] Anika Naidu
*   **Headline**: Cloud Engineer | Cloud & DevOps
*   **Location**: Austin, USA (8.2 Years Experience)
*   **Company**: CRED (Fintech)

> **Summary**: Software engineer with 8.2 years of experience across web, backend, and cloud systems. Strong fundamentals in software development and system design. I've spent most of my career on web and API development — Python/Django and Node.js mostly. I've been keeping up with AI/ML at a self-learner level — taken some online courses, played with the OpenAI and Anthropic APIs, built a small RAG side project — but I haven't done it in a professional capacity yet. Open to roles where I can either deepen my software engineering work or, if the team is open to it, start contributing to ML-adjacent systems.

*   **Skills**: `[CSS (intermediate)]` `[AWS (intermediate)]` `[Airflow (intermediate)]` `[Speech Recognition (intermediate)]` `[FAISS (intermediate)]` `[Six Sigma (intermediate)]` `[Java (intermediate)]` `[ASR (intermediate)]` `[GANs (intermediate)]` `[Apache Beam (beginner)]` `[Data Pipelines (intermediate)]` `[SEO (intermediate)]` `[dbt (intermediate)]`
*   **Experience**:
    *   **Cloud Engineer** at CRED – 32 months *(Current)*
        *   Frontend engineering at a media company. React, TypeScript, and the typical surrounding tooling (Webpack, Jest, Cypress). Built the company's design system from scratch and led the migration from a legacy AngularJS app. Strong on the frontend craft — accessibility, performance, animations — but limited backend exposure.
    *   **Frontend Engineer** at Wayne Enterprises – 19 months
        *   Test automation and QA engineering for a fintech product. Built and maintained the end-to-end test suite using Selenium and pytest, plus the load-testing setup using Locust. Worked closely with developers on testability patterns and with product on acceptance criteria. Recent work has been on shifting test responsibility into the dev team — moving from QA-as-gate to QA-as-coach. Career has been entirely in QA/test engineering.
    *   **Java Developer** at HCL – 46 months
        *   Android mobile development using Java and (more recently) Kotlin at a consumer-app company. Built and maintained multiple production features including the main shopping flow, push notification system, and the offline-first sync layer. Comfortable with the Android framework, Jetpack components, and the typical patterns (MVVM, Hilt, Coroutines). My career has been entirely on mobile so far; interested in expanding into broader backend or platform engineering.
*   **Education**:
    *   **Regional Technical Institute** (tier_4) – M.Sc in Computer Engineering | Grade: 8.73 CGPA
*   **Signals**:
    *   **Expected Salary**: 8.8 - 26.9 LPA (INR) | **Notice Period**: 90 days
    *   **Work Mode**: hybrid | **Profile Completeness**: 87.9% | **GitHub Score**: 30.4

---

#### 👤 [CAND_0099907] Suresh Chowdary
*   **Headline**: Java Developer | Cloud & DevOps
*   **Location**: Mumbai, Maharashtra, India (7.1 Years Experience)
*   **Company**: Acme Corp (Manufacturing)

> **Summary**: Software engineer with 7.1 years of experience across web, backend, and cloud systems. Strong fundamentals in software development and system design. I've spent most of my career on web and API development — Python/Django and Node.js mostly. I've been keeping up with AI/ML at a self-learner level — taken some online courses, played with the OpenAI and Anthropic APIs, built a small RAG side project — but I haven't done it in a professional capacity yet. Open to roles where I can either deepen my software engineering work or, if the team is open to it, start contributing to ML-adjacent systems.

*   **Skills**: `[Databricks (intermediate)]` `[Qdrant (advanced)]` `[GraphQL (beginner)]` `[Spring Boot (beginner)]` `[Accounting (intermediate)]` `[JavaScript (intermediate)]` `[GCP (beginner)]` `[Python (intermediate)]` `[Redis (beginner)]` `[Flask (intermediate)]`
*   **Experience**:
    *   **Java Developer** at Acme Corp – 12 months *(Current)*
        *   Frontend engineering at a media company. React, TypeScript, and the typical surrounding tooling (Webpack, Jest, Cypress). Built the company's design system from scratch and led the migration from a legacy AngularJS app. Strong on the frontend craft — accessibility, performance, animations — but limited backend exposure.
    *   **DevOps Engineer** at Flipkart – 50 months
        *   Frontend engineering at a media company. React, TypeScript, and the typical surrounding tooling (Webpack, Jest, Cypress). Built the company's design system from scratch and led the migration from a legacy AngularJS app. Strong on the frontend craft — accessibility, performance, animations — but limited backend exposure.
    *   **.NET Developer** at Hooli – 22 months
        *   Android mobile development using Java and (more recently) Kotlin at a consumer-app company. Built and maintained multiple production features including the main shopping flow, push notification system, and the offline-first sync layer. Comfortable with the Android framework, Jetpack components, and the typical patterns (MVVM, Hilt, Coroutines). My career has been entirely on mobile so far; interested in expanding into broader backend or platform engineering.
*   **Education**:
    *   **Jadavpur University** (tier_2) – M.S. in Computer Engineering | Grade: 8.50 CGPA
*   **Signals**:
    *   **Expected Salary**: 11.2 - 30.7 LPA (INR) | **Notice Period**: 90 days
    *   **Work Mode**: flexible | **Profile Completeness**: 54.6% | **GitHub Score**: 63.5

---

## 📝 Experiment Conclusion

> [!CAUTION]
> After manually comparing these top results against the target Job Description:
> **These are exactly the profiles the JD told us to outrightly reject or rank low.** Looking at their job titles and summaries, these candidates have no relevant backend ML experience, yet they are ranked in the top results. 
> The semantic quality suffered due to naive threshold-based binarization on a standard bi-encoder (`all-MiniLM-L6-v2`) which was not designed to preserve similarity under binary projection.

---

## 💡 Takeaways

1.  🛠️ **Filter/Remove Skills During Stage 1 Recall**: Listing generic or unrelated skills might dilute the retrieval focus. We should consider filtering or excluding them from the recall embedding payload.
2.  📉 **Naive Binarization Issues**: Simple threshold binarization is not detailed enough to perform high-quality semantic recall on standard models.
3.  ⚖️ **Strict Constraint Importance**: We need to pay special attention to the hard JD constraints, including **work experience**, **job title**, and **years of experience** during the first pass.

---

## 🚀 Future Steps

*   **Reconsider Binarization**: We perhaps need to avoid binarization entirely, or switch to a binarization-optimized model (like `nomic-embed-text-v1.5`) that supports binarization natively without heavy information loss.
*   **Utilize Retrieval Speed**: Since retrieval currently takes less than a second, we have ample headroom to introduce more semantically rich models or hybrid scoring.

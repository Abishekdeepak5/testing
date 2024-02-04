FROM openjdk:11-jre-slim
WORKDIR /app
COPY . /app
RUN javac hello.java
CMD ["java","hello"]
FROM openjdk
WORKDIR /app
COPY . /app
RUN javac hello.java
CMD ["java","hello"]
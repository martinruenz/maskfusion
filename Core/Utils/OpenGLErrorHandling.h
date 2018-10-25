#pragma once

#ifdef NDEBUG
#define checkGLErrors()
#else
#define checkGLErrors() checkGLErrors_(__FILE__, __LINE__)
static void checkGLErrors_(char* file, int line) {
    GLenum err = GL_NO_ERROR;
    while ((err = glGetError()) != GL_NO_ERROR) printf("OpenGL-Error in file %s, line %d: %s (%d)\n", file, line, gluErrorString(err), err);
}

#define checkProgramLog(PRG) checkProgramLog_(PRG, __FILE__, __LINE__)
static void checkProgramLog_(GLuint program, char* file, int line) {
    checkGLErrors_(file,line);

    GLint maxLength = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &maxLength);
    std::vector<GLchar> infoLog(maxLength);
    glGetProgramInfoLog(program, maxLength, &maxLength, &infoLog[0]);
    if(infoLog.size()) printf("Program-log in file %s, line %d: %s", file, line, (const char*)&infoLog[0]);
}

#define checkProgramLinked(PRG) checkProgramLinked_(PRG, __FILE__, __LINE__)
static void checkProgramLinked_(GLuint program, char* file, int line) {
    GLint isLinked = 0;
    glGetProgramiv(program, GL_LINK_STATUS, (int *)&isLinked);
    if (isLinked == GL_FALSE) printf("Program is not linked in file %s, line %d", file, line);
}
#endif

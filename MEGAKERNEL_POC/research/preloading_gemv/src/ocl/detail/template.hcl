#pragma once

#define CAT(X, Y) X##Y
#define TEMPLATE(X, Y) CAT(X, Y)
#ifndef SUFFIX
#define SUFFIX
#endif
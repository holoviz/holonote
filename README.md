# `holonote`

The `holonote` library offers tools to create, edit and persist
annotated regions for [HoloViews](https://holoviews.org/).

![Annotation demo](https://raw.githubusercontent.com/holoviz/holonote/main/doc/_static/assets/demo.gif)

An annotated region marks a region-of-interest that is overlaid on top
of the visual representation of a HoloViews element. Such a region has a
unique identifier as well as additional domain-specific information
associated with it. What `holonote` offers is a flexible way to
interactively work with these visual regions as well as tools to add,
update and delete whatever domain-specific information you want to
associate with them.

There are two primary components currently offered by `holonote`:

1. `Annotators`: These classes offer the ability to interactively define
regions, add associated domain-specific information and then persist
this information to a database. You can interactively add, delete and
update these annotation regions as well as their associated
domain-specific data.

2. `Editors`: These tools are designed to let you interactively
manipulate and edit the visual regions themselves. These regions are in
fact HoloViews elements which means you can use these editors as
generic edit tools for the corresponding HoloViews elements.

By default, annotators automatically persist any annotations you create
to a local SQLite database file. This is a very convenient way to get
started with your annotation task, allowing you to immediate begin the
process of collecting useful information.

For production usage where you are investing significant resources
authoring annotations, it is advisable that you define a custom
connector to a more appropriate database system. You can also use
connectors to load annotations from existing databases of annotation
data.

**Do not use the SQLite connector when deploying to production. It is
  intended for use only during local development. It is not designed to
  be particularly efficient, stable, or secure.**

## Terminology

There are two particularly important pieces of terminology that are used
throughout this library to be aware of:

1. *Regions*: These denote the visual regions-of-interest with a final
visual representation as a HoloViews element. These may be one or two
dimensional, may have different visual representations for the same data
(e.g. a point marker or a crosshair to mark a point in 2D space) and
have different types:

  - *Points*: A single value of interest in an N-dimensional space. For
     instance, on a time axis, this corresponds to a moment in time
     which may be represented visually as a vertical line. On an image,
     this can ve visually represented with a crosshair.
  - *Ranges*: A range is an axis-aligned span between a start and end
     point. For instance, along a time axis, this corresponds to a time interval and
     on an image, this corresponds to a 2D box.

2. *Fields*: These are the domain-specific component of the annotation
data, containing information that is not to be visualized by
`holonote`. These are defined by the user or application and can have
arbitrary types and contents. A complete set of fields together with one
or more regions constitutes an annotation.
